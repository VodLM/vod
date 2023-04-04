from __future__ import annotations

import collections
import functools
import math
from typing import Any, Iterable

import lightning.pytorch as pl
import loguru
import omegaconf
import torch
from rich.progress import track

from raffle_ds_research.core import dataset_builders
from raffle_ds_research.core.ml_models import Ranker
from raffle_ds_research.core.ml_models.monitor import Monitor
from raffle_ds_research.core.workflows import index_manager
from raffle_ds_research.core.workflows.config import CollateConfigs, DataLoaderConfigs, MultiIndexConfig
from raffle_ds_research.core.workflows.utils import (
    _log_eval_metrics,
    _log_retrieval_batch,
    _run_static_evaluation,
    instantiate_retrieval_dataloader,
)


@torch.inference_mode()
def benchmark(
    *,
    ranker: Ranker,
    trainer: pl.Trainer,
    builders: Iterable[dataset_builders.RetrievalBuilder],
    index_cfg: dict | omegaconf.DictConfig | MultiIndexConfig,
    loader_cfg: dict | omegaconf.DictConfig | DataLoaderConfigs,
    collate_cfg: dict | omegaconf.DictConfig | CollateConfigs,
    monitor: Monitor,
) -> dict[str, Any]:
    """Benchmark a ranker on a set of builders."""

    if isinstance(collate_cfg, (dict, omegaconf.DictConfig)):
        collate_cfg = CollateConfigs.parse(collate_cfg)

    if isinstance(loader_cfg, (dict, omegaconf.DictConfig)):
        loader_cfg = DataLoaderConfigs.parse(loader_cfg)

    if isinstance(index_cfg, (dict, omegaconf.DictConfig)):
        index_cfg = MultiIndexConfig(**index_cfg)

    benchmark_data: dict[str, Any] = collections.defaultdict(dict)
    for builder in builders:
        loguru.logger.info(f"Running benchmark for `{builder.name}` ({builder.splits})")
        with index_manager.IndexManager(
            ranker=ranker,
            trainer=trainer,
            builder=builder,
            index_cfg=index_cfg,
            loader_cfg=loader_cfg.predict,
        ) as manager:
            for split in builder.splits:
                # run the static validation & log results
                dataloader = instantiate_retrieval_dataloader(
                    builder=builder,
                    manager=manager,
                    loader_cfg=loader_cfg.eval,
                    collate_config=collate_cfg.static,
                    dset_split=split,
                )
                static_eval_metrics = _run_static_evaluation(
                    loader=track(dataloader, description=f"Evaluating {builder.name}:{split}"),
                    monitor=monitor,
                    on_first_batch=functools.partial(
                        _log_retrieval_batch,
                        tokenizer=builder.tokenizer,
                        gloabl_step=trainer.global_step,
                        max_sections=5,
                        locator=f"{builder.name}/{split}",
                    ),
                )
                _log_eval_metrics(
                    {f"{builder.name}/{k}": v for k, v in static_eval_metrics.items()},
                    trainer=trainer,
                    console=True,
                )
                benchmark_data[builder.name] = static_eval_metrics

    return benchmark_data
