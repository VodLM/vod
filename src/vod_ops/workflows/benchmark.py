import collections
import pathlib

import lightning as L
import numpy as np
import torch
import vod_configs
import vod_dataloaders
import vod_search
import vod_types as vt
from loguru import logger
from vod_models.monitoring import RetrievalMonitor
from vod_ops.utils import helpers, schemas
from vod_tools.misc.config import flatten_dict
from vod_tools.misc.progress import IterProgressBar

DEFAULT_RETRIEVAL_SCORE_KEYS = ["sparse", "dense", "score"]


@torch.no_grad()
def benchmark_retrieval(
    queries: schemas.QueriesWithVectors,
    sections: schemas.SectionsWithVectors,
    *,
    fabric: L.Fabric,
    config: vod_configs.BenchmarkConfig,
    collate_config: vod_configs.RetrievalCollateConfig,
    dataloader_config: vod_configs.DataLoaderConfig,
    cache_dir: pathlib.Path,
    score_keys: None | list[str] = None,
) -> dict[str, float]:
    """Evaluate the cached retriever by benchmarking the scores given by the Realm dataloader."""
    with vod_search.build_hybrid_search_engine(
        sections=sections.sections,
        vectors=sections.vectors,
        configs=sections.search_configs,
        cache_dir=cache_dir,
        dense_enabled=helpers.is_engine_enabled(config.parameters, "dense"),
        sparse_enabled=True,
        serve_on_gpu=config.serve_search_on_gpu,
    ) as master:
        search_client = master.get_client()

        # Instantiate the dataloader
        dataloader = vod_dataloaders.RealmDataloader.factory(
            queries=queries.queries,
            vectors=queries.vectors,
            search_client=search_client,
            collate_config=collate_config,
            parameters=config.parameters,
            **dataloader_config.model_dump(),
        )

        # Wrap the dataloader with lightning
        dataloader = fabric.setup_dataloaders(
            dataloader,
            use_distributed_sampler=True,
            move_to_device=False,  # Everything runs on CPU, no model involved.
        )

        # Run the evaluation
        score_keys = score_keys or DEFAULT_RETRIEVAL_SCORE_KEYS
        monitors = {key: RetrievalMonitor(metrics=config.metrics).to(dtype=torch.float64) for key in score_keys}
        diagnostics = collections.defaultdict(list)

        # Infer the number of steps
        if config.n_max_eval is None:
            num_steps = len(dataloader)
        else:
            eff_batch_size: int = fabric.world_size * dataloader.batch_size  # type: ignore
            num_steps = max(1, -(-config.n_max_eval // eff_batch_size))

        # Callback - test starts
        fabric.call("on_test_start", fabric=fabric, module=None)
        try:
            with IterProgressBar(disable=not fabric.is_global_zero, redirect_stderr=False) as pbar:
                ptask = pbar.add_task("Benchmarking", total=num_steps, info=f"{queries.descriptor}")
                for i, batch in enumerate(dataloader):
                    if i >= num_steps:
                        break

                    # Callback - start of batch
                    fabric.call(
                        "on_test_batch_start",
                        fabric=fabric,
                        batch=batch,
                        module=None,
                        batch_idx=i,
                    )

                    # Compute the metrics
                    for key, monitor in monitors.items():
                        retriever_scores = batch.get(f"section__{key}", None)
                        if retriever_scores is None:
                            continue
                        monitor.update(
                            batch=batch,
                            model_output=vt.ModelOutput(
                                loss=None,
                                retriever_scores=retriever_scores,
                            ),
                        )

                    # Callback - end of batch
                    fabric.call(
                        "on_test_batch_end",
                        fabric=fabric,
                        batch=batch,
                        batch_idx=i,
                        module=None,
                        output=None,
                    )

                    pbar.update(ptask, advance=1)
        except KeyboardInterrupt:
            logger.warning("Benchmark interrupted (KeyboardInterrupt).")

        # Callback - test starts
        fabric.call("on_test_end", fabric=fabric, module=None)

        # aggregate the metrics and the diagnostics
        metrics_dict = {key: monitor.compute() for key, monitor in monitors.items()}
        metrics_dict["diagnostics"] = {k: np.mean(v) for k, v in diagnostics.items()}
        return flatten_dict(metrics_dict, sep="/")
