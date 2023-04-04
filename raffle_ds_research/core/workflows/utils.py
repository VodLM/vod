from __future__ import annotations

import pathlib
from typing import Any, Callable, Iterable, Optional

import lightning.pytorch as pl
import loguru
import rich.console
import torch
import transformers
from rich import terminal_theme

from raffle_ds_research.core.dataset_builders.retrieval_builder import RetrievalBuilder
from raffle_ds_research.core.ml_models.monitor import Monitor
from raffle_ds_research.core.workflows import index_manager
from raffle_ds_research.core.workflows.config import DefaultCollateConfig
from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.utils import loader_config
from raffle_ds_research.utils.pretty import print_metric_groups


def _log_eval_metrics(metrics: dict[str, Any], trainer: pl.Trainer, console: bool = False) -> None:
    for logger in trainer.loggers:
        logger.log_metrics(metrics, step=trainer.global_step)

    if console:
        print_metric_groups(metrics)


def _log_retrieval_batch(
    batch: dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
    gloabl_step: int,
    max_sections: int = 10,
    locator: str = "eval",
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
        html_path = str(pathlib.Path(f"batch-{gloabl_step}.html"))
        console.save_html(html_path, theme=terminal_theme.MONOKAI)

        import wandb

        wandb.log({f"trainer/{locator}/batch": wandb.Html(open(html_path))}, step=gloabl_step)
    except Exception as e:
        loguru.logger.debug(f"Could not log batch to wandb: {e}")


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
    loader_cfg: loader_config.DataLoaderConfig,
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
        **loader_cfg.dict(),
    )
