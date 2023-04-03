from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import loguru
import numpy as np
import rich
import rich.status
import torch
import transformers
from lightning import pytorch as pl
from omegaconf import DictConfig
from rich import terminal_theme
from rich.progress import track

import raffle_ds_research.core.workflows.config
from raffle_ds_research.core.builders import FrankBuilder
from raffle_ds_research.core.builders.retrieval_builder import RetrievalBuilder
from raffle_ds_research.core.ml_models import Ranker
from raffle_ds_research.core.ml_models.monitor import Monitor
from raffle_ds_research.core.workflows import index_manager
from raffle_ds_research.core.workflows.config import DefaultCollateConfig
from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.utils import loader_config
from raffle_ds_research.utils.pretty import print_metric_groups


@dataclasses.dataclass
class TrainWithIndexConfigs:
    train_loader: loader_config.DataLoaderConfig
    eval_loader: loader_config.DataLoaderConfig
    predict_loader: loader_config.DataLoaderConfig
    train_collate: DefaultCollateConfig
    eval_collate: DefaultCollateConfig
    static_eval_collate: DefaultCollateConfig
    indexes: raffle_ds_research.core.workflows.config.MultiIndexConfig

    @classmethod
    def parse(cls, config: DictConfig) -> "TrainWithIndexConfigs":
        # get the dataloader configs
        train_loader_config = loader_config.DataLoaderConfig(**config.loader_configs.train)
        eval_loader_config = loader_config.DataLoaderConfig(**config.loader_configs.eval)
        predict_loader_config = loader_config.DataLoaderConfig(**config.loader_configs.predict)

        # get te collate configs
        train_collate_config = DefaultCollateConfig(**config.collate_configs.train)
        eval_collate_config = DefaultCollateConfig(**config.collate_configs.eval)
        static_eval_collate_config = DefaultCollateConfig(**config.collate_configs.static_eval)

        # set the index configs
        indexes_config = raffle_ds_research.core.workflows.config.MultiIndexConfig(**config.indexes)

        return cls(
            train_loader=train_loader_config,
            eval_loader=eval_loader_config,
            predict_loader=predict_loader_config,
            train_collate=train_collate_config,
            eval_collate=eval_collate_config,
            static_eval_collate=static_eval_collate_config,
            indexes=indexes_config,
        )


def train_with_index_updates(
    ranker: Ranker,
    monitor: Monitor,
    trainer: pl.Trainer,
    builder: FrankBuilder,
    config: TrainWithIndexConfigs | DictConfig,
) -> Ranker:
    """Train a ranker while periodically updating the faiss index."""
    if isinstance(config, DictConfig):
        config = TrainWithIndexConfigs.parse(config)

    total_number_of_steps = trainer.max_steps
    update_steps = _infer_update_steps(total_number_of_steps, config.indexes.update_freq)
    loguru.logger.info(f"Index will be updated at steps: {_pretty_steps(update_steps)}")
    if len(update_steps) == 0:
        raise ValueError("No index update steps were defined.")

    stop_callback = PeriodicStoppingCallback(stop_at=-1)
    trainer.callbacks.append(stop_callback)  # type: ignore

    for period_idx, (start_step, end_step) in enumerate(zip(update_steps[:-1], update_steps[1:])):
        _log_metrics(
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

        # compute question and section vectors, spin up the faiss index.
        with index_manager.IndexManager(
            ranker=ranker,
            trainer=trainer,
            builder=builder,
            config=config.indexes,
            loader_config=config.predict_loader,
            index_step=period_idx,
        ) as manager:
            # run the static validation & log results
            static_val_loader = instantiate_retrieval_dataloader(
                builder=builder,
                manager=manager,
                loader_config=config.eval_loader,
                collate_config=config.static_eval_collate,
                dset_split="validation",
            )
            static_eval_metrics = _run_static_evaluation(
                loader=track(static_val_loader, description="Static validation"),
                monitor=monitor,
                on_first_batch=functools.partial(
                    _log_retrieval_batch,
                    tokenizer=builder.tokenizer,
                    period_idx=period_idx,
                    gloabl_step=trainer.global_step,
                    max_sections=5,
                ),
            )
            _log_metrics(static_eval_metrics, trainer=trainer, console=True)

            # train the model for the given period
            train_loader = instantiate_retrieval_dataloader(
                builder=builder,
                manager=manager,
                loader_config=config.train_loader,
                collate_config=config.train_collate,
                dset_split="train",
            )
            val_loader = instantiate_retrieval_dataloader(
                builder=builder,
                manager=manager,
                loader_config=config.eval_loader,
                collate_config=config.eval_collate,
                dset_split="validation",
            )

            trainer.should_stop = False
            stop_callback.stop_at = end_step
            manager.trainer.fit(
                manager.ranker,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

    return ranker


def _run_static_evaluation(
    loader: Iterable[dict[str, Any]],
    monitor: Monitor,
    on_first_batch: Optional[Callable[[dict[str, Any]], Any]] = None,
) -> dict[str, Any]:
    monitor.reset()
    for i, batch in enumerate(loader):
        if i == 0 and on_first_batch is not None:
            on_first_batch(batch)
        monitor.update_from_retrieval_batch(batch, field="section")
    metrics = monitor.compute(prefix="val/")
    monitor.reset()
    return metrics


def instantiate_retrieval_dataloader(
    *,
    builder: RetrievalBuilder,
    manager: index_manager.IndexManager,
    collate_config: DefaultCollateConfig,
    loader_config: loader_config.DataLoaderConfig,
    dset_split: str,
) -> torch.utils.data.DataLoader:
    full_collate_config = builder.collate_config(
        question_vectors=manager.vectors.dataset.get(dset_split, None),
        clients=manager.clients,
        **collate_config.dict(),
    )
    collate_fn = builder.get_collate_fn(config=full_collate_config)

    return torch.utils.data.DataLoader(
        manager.dataset[dset_split],
        collate_fn=collate_fn,
        **loader_config.dict(),
    )


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


def _log_metrics(metrics: dict[str, Any], trainer: pl.Trainer, console: bool = False) -> None:
    for logger in trainer.loggers:
        logger.log_metrics(metrics, step=trainer.global_step)

    if console:
        print_metric_groups(metrics)


def _log_retrieval_batch(
    batch: dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
    period_idx: int,
    gloabl_step: int,
    max_sections: int = 10,
) -> None:
    try:
        console = rich.console.Console(record=True)
        pipes.pprint_supervised_retrieval_batch(
            batch,
            header="Evaluation batch",
            tokenizer=tokenizer,
            console=console,
            skip_special_tokens=True,
            max_sections=max_sections,
        )
        html_path = str(Path(f"batch-period{period_idx + 1}.html"))
        console.save_html(html_path, theme=terminal_theme.MONOKAI)

        import wandb

        wandb.log({"trainer/eval-batch": wandb.Html(open(html_path))}, step=gloabl_step)
    except Exception as e:
        loguru.logger.debug(f"Could not log batch to wandb: {e}")


class PeriodicStoppingCallback(pl.callbacks.Callback):
    def __init__(self, stop_at: int):
        super().__init__()
        self.stop_at = stop_at

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if trainer.global_step >= self.stop_at:
            trainer.should_stop = True
