import collections
import dataclasses
import json
import pathlib
import typing as typ

import numpy as np
import torch
import transformers
import vod_configs
import vod_dataloaders
import vod_search
from lightning_utilities.core.rank_zero import rank_zero_only
from loguru import logger
from vod_models.monitor import RetrievalMetricCollection
from vod_tools.misc.config import flatten_dict
from vod_tools.misc.progress import IterProgressBar
from vod_workflows.utils import helpers, schemas

_DEFAULT_OUTPUT_KEYS = ["sparse", "dense", "score"]


@dataclasses.dataclass(frozen=True)
class ToDiskConfig:
    """Configuration saving benchmark outputs to disk."""

    logdir: pathlib.Path
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
    max_batches: float | int = 3  # log only the first `n` batches


@torch.no_grad()
@rank_zero_only
def benchmark_retrieval(
    queries: schemas.QueriesWithVectors,
    sections: schemas.SectionsWithVectors,
    *,
    metrics: list[str],
    collate_config: vod_configs.RetrievalCollateConfig,
    dataloader_config: vod_configs.DataLoaderConfig,
    cache_dir: pathlib.Path,
    parameters: None | dict[str, float] = None,
    output_keys: None | list[str] = None,
    serve_search_on_gpu: bool = True,
    n_max: None | int = None,
    to_disk_config: None | ToDiskConfig = None,
) -> dict[str, float]:
    """Run benchmarks on a retrieval task."""
    with vod_search.build_hybrid_search_engine(
        sections=sections.sections,
        vectors=sections.vectors,
        configs=sections.search_configs,
        cache_dir=cache_dir,
        dense_enabled=helpers.is_engine_enabled(parameters, "dense"),
        sparse_enabled=True,
        serve_on_gpu=serve_search_on_gpu,
    ) as master:
        search_client = master.get_client()

        # Instantiate the dataloader
        dataloader = vod_dataloaders.RealmDataloader.factory(
            queries=queries.queries,
            vectors=queries.vectors,
            search_client=search_client,
            collate_config=collate_config,
            parameters=parameters,
            **dataloader_config.model_dump(),
        )

        # Run the evaluation
        output_keys = output_keys or _DEFAULT_OUTPUT_KEYS
        cfg = {"compute_on_cpu": True, "dist_sync_on_step": True, "sync_on_compute": False}
        monitors = {key: RetrievalMetricCollection(metrics=metrics, **cfg) for key in output_keys}
        diagnostics = collections.defaultdict(list)

        try:
            with IterProgressBar() as pbar:
                if n_max is None:  # noqa: SIM108
                    ntotal = len(dataloader)
                else:
                    ntotal = max(1, -(-n_max // dataloader.batch_size))  # type: ignore
                ptask = pbar.add_task(
                    "Benchmarking",
                    total=ntotal,
                    info=f"{queries.descriptor}",
                )
                for i, batch in enumerate(dataloader):
                    if i >= ntotal:
                        break

                    # Log the batch to disk
                    if to_disk_config is not None:
                        _log_retrieval_batch(to_disk_config, batch, batch_idx=i)

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
        metrics_dict = {key: monitor.compute() for key, monitor in monitors.items()}
        metrics_dict["diagnostics"] = {k: np.mean(v) for k, v in diagnostics.items()}
        return flatten_dict(metrics_dict, sep="/")


def _log_retrieval_batch(to_disk_config: ToDiskConfig, batch: dict[str, typ.Any], batch_idx: int) -> None:
    """Log a sampled retrieval batch to disk."""
    if batch_idx >= to_disk_config.max_batches:
        return
    logfile = pathlib.Path(to_disk_config.logdir, f"batch_{batch_idx:05d}.json")
    with logfile.open("w") as f:
        f.write(_safe_json_dumps_batch(batch, tokenizer=to_disk_config.tokenizer, indent=2))


def _safe_json_dumps_batch(
    batch: dict[str, typ.Any],
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
    value: typ.Any,  # noqa: ANN401
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
