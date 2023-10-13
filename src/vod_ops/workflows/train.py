import functools
import multiprocessing as mp
import pathlib
import typing as typ

import lightning as L
import rich
import torch
import vod_configs
import vod_dataloaders
import vod_models
import vod_search
from loguru import logger
from vod_workflows.loops.train import training_loop
from vod_workflows.utils import helpers, schemas
from vod_workflows.utils.trainer_state import TrainerState

K = typ.TypeVar("K")


def spawn_search_and_train(
    state: TrainerState,
    *,
    # ML module & optimizer
    module: vod_models.VodSystem,
    optimizer: torch.optim.Optimizer,
    scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
    # Data
    train_queries: schemas.QueriesWithVectors,
    val_queries: schemas.QueriesWithVectors,
    sections: schemas.SectionsWithVectors,
    # Configs
    collate_config: vod_configs.RetrievalCollateConfig,
    train_dataloader_config: vod_configs.DataLoaderConfig,
    eval_dataloader_config: vod_configs.DataLoaderConfig,
    dl_sampler: None | vod_dataloaders.DlSamplerFactory = None,
    # Utils
    fabric: L.Fabric,
    cache_dir: pathlib.Path,
    serve_on_gpu: bool = False,
) -> TrainerState:
    """Index the sections and train the ranker."""
    barrier_fn = functools.partial(helpers.barrier_fn, fabric=fabric)

    # Make the parameters available between multiple processes using `mp.Manager`
    parameters: typ.MutableMapping = mp.Manager().dict()
    parameters.update(state.get_parameters())

    if fabric.is_global_zero:
        rich.print(
            {
                "train_queries": train_queries,
                "val_queries": val_queries,
                "sections": sections,
                "parameters": dict(parameters),
            }
        )

    barrier_fn("Init search engines..")
    with vod_search.build_hybrid_search_engine(
        sections=sections.sections,
        vectors=sections.vectors,
        configs=sections.search_configs,
        cache_dir=cache_dir,
        dense_enabled=helpers.is_engine_enabled(parameters, "dense"),
        sparse_enabled=True,
        skip_setup=not fabric.is_global_zero,
        barrier_fn=barrier_fn,
        fabric=fabric,
        serve_on_gpu=serve_on_gpu,
    ) as master:
        barrier_fn("Initiating dataloaders")
        search_client = master.get_client()

        # instantiate the dataloader
        logger.debug("Instantiating dataloader..")

        # List of arguments for each dataloader
        shared_args = {
            "search_client": search_client,
            "collate_config": collate_config,
            "parameters": parameters,
            "sampler": dl_sampler,
        }
        train_dl = vod_dataloaders.RealmDataloader.factory(
            queries=train_queries.queries,
            vectors=train_queries.vectors,
            **shared_args,
            **train_dataloader_config.model_dump(),
        )
        val_dl = vod_dataloaders.RealmDataloader.factory(
            queries=val_queries.queries,
            vectors=val_queries.vectors,
            **shared_args,
            **eval_dataloader_config.model_dump(),
        )

        # Patching dataloaders with `Fabric` methods
        train_dl, val_dl = fabric.setup_dataloaders(
            train_dl,
            val_dl,
            use_distributed_sampler=dl_sampler is None,
            move_to_device=True,
        )

        # Train the ranker
        barrier_fn(f"Starting training period {1+state.pidx}")
        trainer_state = training_loop(
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            fabric=fabric,
            train_dl=train_dl,
            val_dl=val_dl,
            parameters=parameters,
        )
        barrier_fn(f"Completed period {1+trainer_state.pidx}")

    return trainer_state
