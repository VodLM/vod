from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Any, Optional, Callable

import loguru
import numpy as np
from lightning import pytorch as pl
from omegaconf import DictConfig

from raffle_ds_research.core.dataset_builders import retrieval_builder
from raffle_ds_research.core.ml_models import Ranker
from raffle_ds_research.core.ml_models.monitor import Monitor
from raffle_ds_research.core.workflows import index_manager
from raffle_ds_research.core.workflows.config import TrainWithIndexConfigs
from raffle_ds_research.core.workflows.retrieval_benchmark import benchmark
from raffle_ds_research.core.workflows.utils import _log_eval_metrics, instantiate_retrieval_dataloader
from raffle_ds_research.tools import cleandir


def _do_nothing(*args, **kwargs):  # noqa: ANN
    pass


def train_with_index_updates(
    ranker_generator: Callable[..., Ranker],
    monitor: Monitor,
    trainer: pl.Trainer,
    builder: retrieval_builder.RetrievalBuilder,
    config: TrainWithIndexConfigs | DictConfig,
    benchmark_builders: Optional[list[retrieval_builder.RetrievalBuilder]] = None,
    benchmark_on_init: bool = False,
) -> Ranker:
    """Train a ranker while periodically updating the faiss index."""
    if isinstance(config, DictConfig):
        config = TrainWithIndexConfigs.parse(config)

    loguru.logger.info("Instantiating the Ranker (init)")
    ranker: Ranker = ranker_generator()

    # Define the index update steps and the `PeriodicStoppingCallback` callback.
    total_number_of_steps = trainer.max_steps
    update_steps = _infer_update_steps(total_number_of_steps, config.indexes.update_freq)
    loguru.logger.info(f"Index will be updated at steps: {_pretty_steps(update_steps)}")
    if len(update_steps) == 0:
        raise ValueError("No index update steps were defined.")

    stop_callback = PeriodicStoppingCallback(stop_at=-1)
    trainer.callbacks.append(stop_callback)  # type: ignore

    # setup the distributed environment, if any.
    # This is done automatically by the trainer when lauching a task (hacky, but working).
    # Might be unnecessary in future versions of lightning.
    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(_do_nothing)

    # Train the model for each period
    for period_idx, (start_step, end_step) in enumerate(zip(update_steps[:-1], update_steps[1:])):
        _log_eval_metrics(
            {
                "trainer/period": float(period_idx + 1),
                "trainer/faiss_weight": config.indexes.faiss.get_weight(period_idx),
                "trainer/bm25_weight": config.indexes.bm25.get_weight(period_idx),
            },
            trainer=trainer,
        )
        loguru.logger.info(
            f"Training period {period_idx + 1}/{len(update_steps) - 1} (step {start_step} -> {end_step})"
        )

        # wrap everything in a temporary directory to avoid filling up the disk.
        # The temporary directory will be deleted at the end of each period.
        with cleandir.CleanableDirectory(
            pathlib.Path(config.sys.cache_dir, f"cached-preds-{period_idx}"),
            delete_existing=True,
        ) as tmp_dir:
            builder()
            # run the benchmarks on init
            if benchmark_builders is not None and (period_idx > 0 or benchmark_on_init):
                loguru.logger.info(f"Starting training epoch {1+period_idx}. Running benchmarks...")
                benchmark(
                    ranker=ranker,
                    trainer=trainer,
                    builders=benchmark_builders,
                    loader_cfg=config.dataloaders,
                    collate_cfg=config.collates,
                    index_cfg=config.indexes,
                    monitor=monitor,
                )

            # compute question and section vectors, spin up the faiss index.
            with index_manager.IndexManager(
                ranker=ranker,
                trainer=trainer,
                builder=builder,
                index_cfg=config.indexes,
                loader_cfg=config.dataloaders.predict,
                cache_dir=Path(tmp_dir),
            ) as manager:
                trainer.strategy.barrier(f"index_manager_{period_idx}")
                # train the model for the given period
                train_loader = instantiate_retrieval_dataloader(
                    builder=builder,
                    manager=manager,
                    loader_cfg=config.dataloaders.train,
                    collate_config=config.collates.train,
                    dset_split="train",
                    use_sampler=True,
                )
                val_loader = instantiate_retrieval_dataloader(
                    builder=builder,
                    manager=manager,
                    loader_cfg=config.dataloaders.eval,
                    collate_config=config.collates.eval,
                    dset_split="validation",
                    use_sampler=True,
                )

                # instantiate the ranker for this period (except for the first one)
                if period_idx > 0 and config.indexes.reset_model:
                    loguru.logger.info(f"Instantiating the Ranker (period={1 + period_idx})")
                    trainable_ranker: Ranker = ranker_generator()
                else:
                    trainable_ranker = ranker

                trainer.should_stop = False
                stop_callback.stop_at = end_step
                manager.trainer.fit(
                    trainable_ranker,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )

                ranker = trainable_ranker

            loguru.logger.info(f"Training period {1 + period_idx} completed.")

    # run the final benchmarks
    if benchmark_builders is not None:
        loguru.logger.info("Training completed. Running final benchmarks...")
        benchmark(
            ranker=ranker,
            trainer=trainer,
            builders=benchmark_builders,
            loader_cfg=config.dataloaders,
            collate_cfg=config.collates,
            index_cfg=config.indexes,
            monitor=monitor,
        )

    return ranker


def _infer_update_steps(total_number_of_steps: int, update_freq: int | list[int]) -> list[int]:
    if isinstance(update_freq, int):
        steps = [int(x) for x in np.arange(0, total_number_of_steps, update_freq)]
    elif isinstance(update_freq, list):
        if update_freq[0] != 0:
            update_freq = [0] + update_freq
        if update_freq[-1] == total_number_of_steps:
            update_freq = update_freq[:-1]
        steps = update_freq
    else:
        raise TypeError(f"Invalid type for `update_freq`: {type(update_freq)}")

    return steps + [total_number_of_steps]


def _pretty_steps(steps: list[int]) -> str:
    steps = steps[:-1]
    if len(steps) > 6:
        return f"[{steps[0]}, {steps[1]}, {steps[2]}, {steps[3]}, {steps[4]} ... {steps[-1]}]"
    else:
        return str(steps)


class PeriodicStoppingCallback(pl.callbacks.Callback):
    """Stops the training after a given number of steps."""

    def __init__(self, stop_at: int):
        super().__init__()
        self.stop_at = stop_at

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,  # noqa: ANN401
        batch: Any,  # noqa: ANN401
        batch_idx: int,
    ) -> None:
        """Called when the training batch ends."""
        if trainer.global_step >= self.stop_at:
            trainer.should_stop = True
