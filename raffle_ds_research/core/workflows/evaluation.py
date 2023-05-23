from __future__ import annotations

import pathlib
from typing import Iterable, Optional

import torch
from lightning.pytorch import utilities as pl_utils
from loguru import logger
from rich.progress import track

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core.mechanics import dataset_factory, search_engine
from raffle_ds_research.core.ml.monitor import RetrievalMetricCollection
from raffle_ds_research.core.workflows.precompute import PrecomputedDsetVectors
from raffle_ds_research.core.workflows.utils import support
from raffle_ds_research.utils.config import flatten_dict

_DEFAULT_OUTPUT_KEYS = ["faiss", "bm25", "score"]


@torch.no_grad()
@pl_utils.rank_zero_only
def benchmark(
    factory: dataset_factory.DatasetFactory,
    *,
    vectors: None | PrecomputedDsetVectors,
    metrics: Iterable[str],
    search_config: core_config.SearchConfig,
    collate_config: core_config.RetrievalCollateConfig,
    dataloader_config: core_config.DataLoaderConfig,
    cache_dir: pathlib.Path,
    parameters: Optional[dict[str, float]] = None,
    output_keys: Optional[list[str]] = None,
    serve_on_gpu: bool = True,
) -> dict[str, float]:
    """Run benchmarks on a retrieval task."""
    with search_engine.build_search_engine(
        sections=factory.get_sections(),  # type: ignore
        vectors=support.maybe_as_lazy_array(vectors.sections),
        config=search_config,
        cache_dir=cache_dir,
        faiss_enabled=support.is_engine_enabled(parameters, "faiss"),
        bm25_enabled=support.is_engine_enabled(parameters, "bm25"),
        serve_on_gpu=serve_on_gpu,
    ) as master:
        search_client = master.get_client()

        # instantiate the dataloader
        dataloader = support.instantiate_retrieval_dataloader(
            questions=support.DsetWithVectors.cast(
                data=factory.get_qa_split(),
                vectors=vectors.questions,
            ),
            sections=support.DsetWithVectors.cast(
                data=factory.get_sections(),
                vectors=vectors.sections,
            ),
            tokenizer=factory.config.tokenizer,
            search_client=search_client,
            collate_config=collate_config,
            dataloader_config=dataloader_config,
            parameters=parameters,
            cache_dir=cache_dir,
            barrier_fn=logger.debug,
            rank=0,
        )

        # run the evaluation
        output_keys = output_keys or _DEFAULT_OUTPUT_KEYS
        cfg = {"compute_on_cpu": True, "dist_sync_on_step": True, "sync_on_compute": False}
        monitors = {key: RetrievalMetricCollection(metrics=metrics, **cfg) for key in output_keys}
        for batch in track(
            dataloader,
            description=f"Benchmarking `{factory.config.name}:{factory.config.split}`",
            total=len(dataloader),
        ):
            target = batch["section.label"]
            for key, monitor in monitors.items():
                preds = batch.get(f"section.{key}", None)
                if preds is None:
                    continue
                monitor.update(preds, target)

        metrics = {key: monitor.compute() for key, monitor in monitors.items()}
        return flatten_dict(metrics, sep="/")
