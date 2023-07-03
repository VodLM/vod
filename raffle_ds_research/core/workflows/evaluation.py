from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
from typing import Any, Iterable, Optional

import numpy as np
import torch
import transformers
from lightning.pytorch import utilities as pl_utils
from loguru import logger

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core.mechanics import dataset_factory, search_engine
from raffle_ds_research.core.ml.monitor import RetrievalMetricCollection
from raffle_ds_research.core.workflows.precompute import PrecomputedDsetVectors
from raffle_ds_research.core.workflows.utils import support
from raffle_ds_research.tools.utils.progress import IterProgressBar
from raffle_ds_research.utils.config import flatten_dict

_DEFAULT_OUTPUT_KEYS = ["faiss", "bm25", "score"]


@dataclasses.dataclass(frozen=True)
class ToDiskConfig:
    """Configuration saving benchmark outputs to disk."""

    logdir: pathlib.Path
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast


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
    n_max: Optional[int] = None,
    to_disk_config: Optional[ToDiskConfig] = None,
) -> dict[str, float]:
    """Run benchmarks on a retrieval task."""
    with search_engine.build_search_engine(
        sections=factory.get_sections(),  # type: ignore
        vectors=support.maybe_as_lazy_array(vectors.sections) if vectors is not None else None,
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
                vectors=vectors.questions if vectors is not None else None,
            ),
            sections=support.DsetWithVectors.cast(
                data=factory.get_sections(),
                vectors=vectors.sections if vectors is not None else None,
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
        diagnostics = collections.defaultdict(list)
        with IterProgressBar() as pbar:
            ntotal = len(dataloader) if n_max is None else max(1, -(-n_max // dataloader.batch_size))  # type: ignore
            ptask = pbar.add_task(
                "Benchmarking",
                total=ntotal,
                info=f"{factory.config.name}:{factory.config.split}",
            )
            for i, batch in enumerate(dataloader):
                if i >= ntotal:
                    break

                # log the batch
                if to_disk_config is not None:
                    logfile = pathlib.Path(to_disk_config.logdir, f"batch_{i:05d}.json")
                    with logfile.open("w") as f:
                        f.write(_safe_json_dumps_batch(batch, tokenizer=to_disk_config.tokenizer, indent=2))

                # gather diagnostics
                diagnostics["n_sections"].append(batch["section.score"].shape[-1])
                for k, v in batch.items():
                    if k.startswith("diagnostics."):
                        diagnostics[k.replace("diagnostics.", "")].append(v)

                # compute and collect the metrics
                target = batch["section.label"]
                for key, monitor in monitors.items():
                    preds = batch.get(f"section.{key}", None)
                    if preds is None:
                        continue
                    monitor.update(preds, target)

                pbar.update(ptask, advance=1)

        # aggregate the metrics and the diagnostics
        metrics = {key: monitor.compute() for key, monitor in monitors.items()}
        metrics["diagnostics"] = {k: np.mean(v) for k, v in diagnostics.items()}
        return flatten_dict(metrics, sep="/")


def _safe_json_dumps_batch(
    batch: dict[str, Any],
    *,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    **kwargs,  # noqa: ANN003
) -> str:
    """Cast a value to a JSON-safe type and serialize it."""
    try:
        return json.dumps(
            {
                k: _safe_json_cast(
                    v,
                    key=k,
                    tokenizer=tokenizer,
                )
                for k, v in batch.items()
                if not k.endswith(".attention_mask")
            },
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to serialize batch: {e}")
        return json.dumps({"error": str(e)})


def _safe_json_cast(
    value: Any,  # noqa: ANN401
    key: str,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
) -> str | int | list | dict:  # noqa: ANN401
    """Cast a value to a JSON-safe type."""
    if key.endswith(".input_ids"):
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise TypeError(f"Expected a tensor, got {type(value)}")
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        if value.ndim == 3:  # noqa: PLR2004
            return [tokenizer.batch_decode(v, skip_special_tokens=True) for v in value]
        if value.ndim == 2:  # noqa: PLR2004
            return tokenizer.batch_decode(value, skip_special_tokens=True)
        if value.ndim == 1:
            return tokenizer.decode(value, skip_special_tokens=True)

        raise ValueError(f"Expected a tensor of rank 1, 2 or 3, got {value.ndim}")

    if isinstance(value, torch.Tensor):
        return value.cpu().detach().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()

    return value
