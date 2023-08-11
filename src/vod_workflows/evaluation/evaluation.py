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
from vod_models.monitor import RetrievalMetricCollection
from vod_tools import dstruct
from vod_tools.misc.config import flatten_dict
from vod_tools.misc.progress import IterProgressBar
from vod_workflows.utils import helpers

from src import vod_configs, vod_datasets, vod_search

_DEFAULT_OUTPUT_KEYS = ["sparse", "dense", "score"]


@dataclasses.dataclass(frozen=True)
class ToDiskConfig:
    """Configuration saving benchmark outputs to disk."""

    logdir: pathlib.Path
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast


@torch.no_grad()
@pl_utils.rank_zero_only
def benchmark(
    task: helpers.RetrievalTask,
    *,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    metrics: Iterable[str],
    collate_config: vod_configs.RetrievalCollateConfig,
    dataloader_config: vod_configs.DataLoaderConfig,
    cache_dir: pathlib.Path,
    parameters: Optional[dict[str, float]] = None,
    output_keys: Optional[list[str]] = None,
    serve_on_gpu: bool = True,
    n_max: Optional[int] = None,
    to_disk_config: Optional[ToDiskConfig] = None,
) -> dict[str, float]:
    """Run benchmarks on a retrieval task."""
    with vod_search.build_hybrid_search_engine(
        shard_names=[cfg.descriptor for cfg in task.sections],
        sections=[vod_datasets.load_sections(cfg) for cfg in task.sections],  # type: ignore
        vectors=[dstruct.as_lazy_array(task.vectors[d]) for d in task.sections]
        if task.vectors
        else None,  # type: ignore
        configs=[cfg.search for cfg in task.sections],  # type: ignore
        cache_dir=cache_dir,
        dense_enabled=helpers.is_engine_enabled(parameters, "dense"),
        sparse_enabled=True,
        serve_on_gpu=serve_on_gpu,
    ) as master:
        search_client = master.get_client()

        # Instantiate the dataloader
        dataloader = helpers.instantiate_retrieval_dataloader(
            queries=helpers.DsetWithVectors.cast(
                data=[vod_datasets.load_queries(cfg) for cfg in task.queries],
                vectors=[task.vectors[d] for d in task.queries] if task.vectors else None,
            ),
            sections=helpers.DsetWithVectors.cast(
                data=[vod_datasets.load_sections(cfg) for cfg in task.sections],
                vectors=[task.vectors[d] for d in task.sections] if task.vectors else None,
            ),
            tokenizer=tokenizer,
            search_client=search_client,
            collate_config=collate_config,
            dataloader_config=dataloader_config,
            parameters=parameters,
        )

        # Run the evaluation
        output_keys = output_keys or _DEFAULT_OUTPUT_KEYS
        cfg = {"compute_on_cpu": True, "dist_sync_on_step": True, "sync_on_compute": False}
        monitors = {key: RetrievalMetricCollection(metrics=metrics, **cfg) for key in output_keys}
        diagnostics = collections.defaultdict(list)
        queries_descriptior = "+".join(cfg.descriptor for cfg in task.queries)

        try:
            with IterProgressBar() as pbar:
                if n_max is None:  # noqa: SIM108
                    ntotal = len(dataloader)
                else:
                    ntotal = max(1, -(-n_max // dataloader.batch_size))  # type: ignore
                ptask = pbar.add_task(
                    "Benchmarking",
                    total=ntotal,
                    info=f"{queries_descriptior}",
                )
                for i, batch in enumerate(dataloader):
                    if i >= ntotal:
                        break

                    # Log the batch to disk
                    if to_disk_config is not None:
                        logfile = pathlib.Path(to_disk_config.logdir, f"batch_{i:05d}.json")
                        with logfile.open("w") as f:
                            f.write(_safe_json_dumps_batch(batch, tokenizer=to_disk_config.tokenizer, indent=2))

                    # Gather the diagnostics
                    diagnostics["n_sections"].append(batch["section.score"].shape[-1])
                    for k, v in batch.items():
                        if k.startswith("diagnostics."):
                            diagnostics[k.replace("diagnostics.", "")].append(v)

                    # Compute and collect the metrics
                    target = batch["section.label"]
                    for key, monitor in monitors.items():
                        preds = batch.get(f"section.{key}", None)
                        if preds is None:
                            continue
                        monitor.update(preds, target)

                    pbar.update(ptask, advance=1)
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted (KeyboardInterrupt).")

        # aggregate the metrics and the diagnostics
        metrics = {key: monitor.compute() for key, monitor in monitors.items()}
        metrics["diagnostics"] = {k: np.mean(v) for k, v in diagnostics.items()}  # type: ignore
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
