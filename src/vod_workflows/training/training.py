import functools
import multiprocessing as mp
import pathlib
import typing as typ

import lightning as L
import rich
import torch
import transformers
import vod_configs
import vod_dataloaders
import vod_models
import vod_search
from loguru import logger
from vod_tools.ts_factory.ts_factory import TensorStoreFactory
from vod_workflows.utils import helpers

from .callbacks import OnFirstBatchCallback
from .loop import training_loop

K = typ.TypeVar("K")


def index_and_train(
    *,
    ranker: vod_models.Ranker,
    optimizer: torch.optim.Optimizer,
    scheduler: None | torch.optim.lr_scheduler._LRScheduler = None,
    trainer_state: helpers.TrainerState,
    fabric: L.Fabric,
    train_queries: list[vod_configs.QueriesDatasetConfig],
    val_queries: list[vod_configs.QueriesDatasetConfig],
    sections: list[vod_configs.SectionsDatasetConfig],
    vectors: None | dict[vod_configs.BaseDatasetConfig, TensorStoreFactory],
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    collate_config: vod_configs.RetrievalCollateConfig,
    train_dataloader_config: vod_configs.DataLoaderConfig,
    eval_dataloader_config: vod_configs.DataLoaderConfig,
    dl_sampler: None | vod_dataloaders.DlSamplerFactory = None,
    cache_dir: pathlib.Path,
    serve_on_gpu: bool = False,
    checkpoint_path: None | str = None,
    on_first_batch_fn: None | OnFirstBatchCallback = None,
    pbar_keys: None | list[str] = None,
) -> helpers.TrainerState:
    """Index the sections and train the ranker."""
    barrier_fn = functools.partial(helpers.barrier_fn, fabric=fabric)

    # Make the parameters available between multiple processes using `mp.Manager`
    parameters: typ.MutableMapping = mp.Manager().dict()
    parameters.update(trainer_state.get_parameters())

    # Wrap queries with vectorss
    train_queries_ = helpers.QueriesWithVectors.from_configs(
        queries=train_queries,
        vectors=vectors,
    )
    val_queries_ = helpers.QueriesWithVectors.from_configs(
        queries=val_queries,
        vectors=vectors,
    )
    sections_ = helpers.SectionsWithVectors.from_configs(
        sections=sections,
        vectors=vectors,
    )

    if fabric.is_global_zero:
        rich.print(
            {
                "train_queries": train_queries_,
                "val_queries": val_queries_,
                "sections": sections_,
                "parameters": dict(parameters),
            }
        )

    barrier_fn("Init search engines..")
    with vod_search.build_hybrid_search_engine(
        sections=sections_.sections,
        vectors=sections_.vectors,
        configs=sections_.search_configs,
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
            "sections": sections_.sections,
            "tokenizer": tokenizer,
            "search_client": search_client,
            "collate_config": collate_config,
            "parameters": parameters,
            "sampler": dl_sampler,
        }
        train_dl = vod_dataloaders.RealmDataloader.factory(
            queries=train_queries_.queries,
            **shared_args,
            **train_dataloader_config.model_dump(),
        )
        val_dl = vod_dataloaders.RealmDataloader.factory(
            queries=val_queries_.queries,
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
        barrier_fn(f"Starting training period {1+trainer_state.pidx}")
        trainer_state = training_loop(
            ranker=ranker,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
            fabric=fabric,
            train_dl=train_dl,
            val_dl=val_dl,
            checkpoint_path=checkpoint_path,
            on_first_batch_fn=on_first_batch_fn,
            pbar_keys=pbar_keys,
            parameters=parameters,
        )
        barrier_fn(f"Completed period {1+trainer_state.pidx}")

    return trainer_state
