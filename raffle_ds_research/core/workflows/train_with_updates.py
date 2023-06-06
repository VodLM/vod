from __future__ import annotations

import functools
import pathlib
import sys
from typing import Any, Callable, Iterable

import lightning as L  # noqa: N812
import numpy as np
import omegaconf
from loguru import logger
from torch.utils import data as torch_data

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core.mechanics import dataset_factory
from raffle_ds_research.core.ml import Ranker
from raffle_ds_research.core.workflows import evaluation, precompute, training
from raffle_ds_research.core.workflows.utils import logging, support
from raffle_ds_research.tools import caching, pipes


def train_with_index_updates(  # noqa: C901, PLR0915
    *,
    trainer: L.Trainer,
    ranker_generator: Callable[..., Ranker],
    config: core_config.TrainWithIndexUpdatesConfigs | omegaconf.DictConfig,
) -> Ranker:
    """Train a ranking model while periodically updating the index."""
    if isinstance(config, omegaconf.DictConfig):
        config = core_config.TrainWithIndexUpdatesConfigs.parse(config)

    logger.info("Instantiating the Ranker (init.)")
    ranker: Ranker = ranker_generator()
    logger.info(f"Encoder output shape: `{ranker.get_output_shape()}`")

    # Define the index update steps and the `PeriodicStoppingCallback` callback.
    update_steps = _infer_update_steps(trainer.max_steps, config.schedule.period)
    logger.info(f"Index will be updated at steps: {_pretty_steps(update_steps)}")
    if len(update_steps) == 0:
        raise ValueError("No index update steps were defined.")

    # Setup the distributed environment, skipping this step would break the `barrier()`
    logger.info("Setting up environment...")
    _hacky_setup_distributed_env(trainer=trainer, config=config, ranker=ranker)
    # trainer.strategy.setup_environment()

    barrier = functools.partial(support._barrier_fn, trainer=trainer)
    barrier("Setup done.")
    barrier("Training starting..")

    # configure logging
    _configure_logger(trainer)

    # Train the model for each period
    update_steps = update_steps + [None]
    for period_idx, (start_step, end_step) in enumerate(zip(update_steps[:-1], update_steps[1:])):
        parameters = config.schedule.get_parameters(trainer.global_step)
        logging.log(
            {
                "trainer/period": float(period_idx + 1),
                **{f"parameter/{k}": v for k, v in parameters.items()},
            },
            trainer=trainer,
        )
        logger.info(f"Starting period {period_idx + 1}/{len(update_steps) - 1} (step {start_step} -> {end_step})")

        # wrap everything in a temporary directory to avoid filling up the disk.
        # The temporary directory will be deleted at the end of each period except the first one
        # as dataset vectors won't change when using the same seed/model.
        with caching.CacheManager(
            pathlib.Path(config.sys.cache_dir, f"train-with-updates-{period_idx}"),
            delete_existing=False,
            persist=period_idx == 0,  # keep the cache during the first period
        ) as cache_dir:
            factories = _get_dset_factories(config.dataset.get("all"), config=config.dataset)

            # pre-process all datasets on global zero, this is done implicitely when loading a dataset
            if trainer.is_global_zero:
                for key, factory in factories.items():
                    logger.debug(f"Pre-processing `{key}` ...")
                    factory.get_qa_split()
                    factory.get_sections()
            barrier("Pre-processing done.")

            # pre-compute the vectors for each dataset, this is deactivated when faiss is not in use
            if support.is_engine_enabled(parameters, "faiss"):
                vectors = precompute.compute_vectors(
                    factories,
                    ranker=ranker,
                    trainer=trainer,
                    cache_dir=cache_dir,
                    dataset_config=config.dataset,
                    collate_config=config.collates.predict,
                    dataloader_config=config.dataloaders.predict,
                )
            else:
                logger.info("Faiss engine is disabled. Skipping vector pre-computation.")
                vectors = {dset: None for dset in factories}

            # benchmark the ranker on each dataset separately
            barrier("running benchmarks..")
            ranker.cpu()  # <- free up the GPU memory so it can be used to build faiss
            if trainer.is_global_zero and period_idx > 0 or config.schedule.benchmark_on_init:
                logger.info(f"Running benchmarks ... (period={1+period_idx})")
                for j, dset in enumerate(config.dataset.benchmark):
                    logger.info(
                        f"{1+j}/{len(config.dataset.benchmark)} - Benchmarking `{dset.name}:{dset.split_alias}` ..."
                    )
                    metrics = evaluation.benchmark(
                        factory=factories[dset],
                        vectors=vectors[dset],
                        metrics=config.dataset.metrics,
                        search_config=config.search.benchmark,
                        collate_config=config.collates.benchmark,
                        dataloader_config=config.dataloaders.eval,
                        cache_dir=cache_dir,
                        parameters=parameters,
                        serve_on_gpu=True,
                    )
                    if metrics is not None:
                        logging.log(
                            {f"{dset.name}/{dset.split_alias}/{k}": v for k, v in metrics.items()},
                            trainer=trainer,
                            console=True,
                            header=f"{dset.name}:{dset.split_alias} - Period {period_idx + 1}",
                        )

            if end_step is None:
                # If there is no defined `end_step`, we reached the end of the training
                continue

            # potentially re-init the ranker
            barrier("Setting up ranker..")
            if config.schedule.reset_model_on_period_start:
                logger.info(f"Re-initializing the Ranker ... (period={1+period_idx}))")
                ranker = ranker_generator()

            # training for the current period.
            # We use a `StopAtTrainer` to stop the training at the end of the current period (max `end_step`).
            logger.info(f"Starting training period {1+period_idx}")
            with WithCallbacks(
                trainer,
                callbacks=[
                    StopAtCallback(end_step),
                    OnFirstBatchCallback(
                        functools.partial(logging.log_retrieval_batch, tokenizer=config.dataset.tokenizer)
                    ),
                    OnFirstBatchCallback(
                        functools.partial(
                            pipes.pprint_batch,
                            header=f"Train batch - period = {1+period_idx}",
                            footer="\n\n",  # <- force a new line # os.get_terminal_size().columns +
                        )
                    ),
                ],  # type: ignore
            ):
                ranker.cpu()  # <- free up the GPU memory so it can be used to build faiss
                ranker = training.index_and_train(
                    ranker=ranker,
                    trainer=trainer,
                    vectors=vectors,
                    train_factories=_get_dset_factories(config.dataset.get("train"), config=config.dataset),
                    val_factories=_get_dset_factories(config.dataset.get("validation"), config=config.dataset),
                    tokenizer=config.dataset.tokenizer,
                    search_config=config.search.training,
                    collate_config=config.collates.train,
                    train_dataloader_config=config.dataloaders.train,
                    eval_dataloader_config=config.dataloaders.eval,
                    dl_sampler=config.dl_sampler,
                    cache_dir=cache_dir,
                    parameters=parameters,
                    serve_on_gpu=False,
                )
            logger.success(f"Completed training period {1+period_idx}")
            barrier("Period completed.")

    return ranker


def _hacky_setup_distributed_env(
    *, trainer: L.Trainer, ranker: Ranker, config: core_config.TrainWithIndexUpdatesConfigs
) -> None:
    """Setup the distributed environment, if any.

    This is done automatically by the trainer when lauching a task (hacky, but working).
    Might be unnecessary in future versions of lightning.
    If not done, `barrier()` will not work as expected.
    """
    factory = next(iter(_get_dset_factories(config.dataset.get("validation"), config=config.dataset).values()))
    dset = factory(what="question").select(range(2 * trainer.num_devices))
    collate_fn = precompute.init_predict_collate_fn(
        config.collates.benchmark,
        field="question",
        tokenizer=config.dataset.tokenizer,
    )
    dl = torch_data.DataLoader(
        dset,  # type: ignore
        collate_fn=collate_fn,
        **config.dataloaders.predict.dict(),
    )
    trainer.predict(model=ranker, dataloaders=dl)


def _get_dset_factories(
    dsets: Iterable[core_config.NamedDset],
    config: core_config.MultiDatasetFactoryConfig,
) -> dict[core_config.NamedDset, dataset_factory.DatasetFactory]:
    return {
        dset_cfg: dataset_factory.DatasetFactory.from_config(
            config.dataset_factory_config(dset_cfg),
        )
        for dset_cfg in dsets
    }


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


def _pretty_steps(steps: list[int], max_steps: int = 6) -> str:
    steps = steps[:-1]
    if len(steps) > max_steps:
        return f"[{steps[0]}, {steps[1]}, {steps[2]}, {steps[3]}, {steps[4]} ... {steps[-1]}]"

    return str(steps)


class WithCallbacks:
    """Context manager to temporarily add a `StopAtCallback` to a trainer."""

    def __init__(self, trainer: L.Trainer, callbacks: list[L.Callback]) -> None:
        self.trainer = trainer
        self.callbacks = callbacks

    def __enter__(self) -> L.Trainer:  # noqa: D105
        self.trainer.callbacks.extend(self.callbacks)  # type: ignore
        return self.trainer

    def __exit__(self, *args, **kwargs) -> None:  # noqa: ANN, D105, ANN
        for callback in self.callbacks:
            self.trainer.callbacks.remove(callback)  # type: ignore


class StopAtCallback(L.Callback):
    """Stops the training after a given number of steps."""

    def __init__(self, stop_at: int):
        super().__init__()
        self.stop_at = stop_at

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # noqa: ARG002
        """Called when the training begins."""
        trainer.should_stop = False

    def on_train_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int  # noqa: ANN, ARG
    ) -> None:
        """Called when the training batch ends."""
        if trainer.global_step >= self.stop_at:
            trainer.should_stop = True


class OnFirstBatchCallback(L.Callback):
    """Log the first batch of the training."""

    def __init__(self, fn: Callable[[dict[str, Any]], Any]) -> None:
        self.fn = fn

    def on_train_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int  # noqa: ANN, ARG
    ) -> None:
        """Called when the training batch starts."""
        if batch_idx == 0 and trainer.global_step == 0 and trainer.is_global_zero:
            self.fn(batch)


def _configure_logger(trainer: L.Trainer) -> None:
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<yellow>rank</yellow>:<yellow>{extra[rank]}/{extra[world_size]}</yellow> - <level>{message}</level>"
    )
    if trainer.world_size == 1:
        return

    logger.configure(extra={"rank": 1 + trainer.global_rank, "world_size": trainer.world_size})  # Default values
    logger.remove()
    logger.add(sys.stderr, format=logger_format)
