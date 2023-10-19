import functools
import pathlib
import typing as typ

import lightning as L
import torch
import vod_configs
import vod_datasets
import vod_models
import vod_types as vt
from lightning.fabric.wrappers import is_wrapped
from loguru import logger
from vod_exps.structconf import Experiment
from vod_ops.utils import TrainerState, helpers, io, logging, schemas
from vod_ops.workflows.benchmark import benchmark_retrieval
from vod_ops.workflows.compute import compute_vectors
from vod_ops.workflows.train import spawn_search_and_train
from vod_tools import cache_manager


def periodic_training(
    *,
    module: vod_models.VodSystem,
    optimizer: torch.optim.Optimizer,
    scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
    fabric: L.Fabric,
    config: Experiment,
    resume_from_checkpoint: None | str = None,
) -> torch.nn.Module:
    """Train a ranking model while periodically updating the index."""
    barrier = functools.partial(helpers.barrier_fn, fabric=fabric)
    state = TrainerState.from_config(config=config.trainer)

    # Check that models and optimizers are wrapped
    if not is_wrapped(module) or not is_wrapped(optimizer):
        raise ValueError("Model and optimizer must be wrapped using `Fabric.setup()`")

    # Load an existing checkpoint
    if resume_from_checkpoint is not None:
        logger.info(f"Loading training state from `{config.trainer.checkpoint_path}`")
        io.load_training_state(
            checkpoint_path=resume_from_checkpoint,
            fabric=fabric,
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=state,
        )

    # Train the model for each period
    barrier("Training starting..")
    fabric.call("on_fit_start", fabric=fabric, module=module)
    while not state.completed:
        # Wrap everything in a temporary directory to avoid filling up the disk.
        # The temporary directory will be deleted at the end of each period except the first one
        # as dataset vectors won't change when using the same seed/model.
        with cache_manager.CacheManager(
            pathlib.Path(config.sys.cache_dir, f"train-with-updates-{state.pidx}"),
            delete_existing=False,
            persist=state.pidx == 0,  # keep the cache for the first period
        ) as cache_dir:
            # pre-process all datasets on rank zero on each node,
            # loading & preprocessing happens implicitely when loading a dataset
            if state.pidx == 0 and fabric.local_rank == 0:
                for cfg in config.get_dataset_configs(split="all", what="all"):
                    vod_datasets.load_dataset(cfg)
            barrier("Pre-processing done.")

            # Run the benchmarks before each training period, except the first one
            if state.pidx > 0 or config.benchmark.on_init:
                _run_benchmarks(
                    module=module,
                    fabric=fabric,
                    config=config,
                    state=state,
                    cache_dir=cache_dir,
                    barrier=barrier,
                )

            # Training the model for the current period.
            period_idx = state.pidx
            logger.info(f"Starting training period {1+period_idx}")
            state = _train_for_period(
                state=state,
                module=module,
                optimizer=optimizer,
                scheduler=scheduler,
                fabric=fabric,
                config=config,
                cache_dir=cache_dir,
            )
            logger.success(f"Completed training period {1+period_idx}")
            barrier("Period completed.")

    fabric.call("on_fit_end", fabric=fabric, module=module)
    return module


def _compute_all_vectors(
    *,
    dset_configs: typ.Iterable[vod_configs.DatasetConfig],
    fabric: L.Fabric,
    module: vod_models.VodSystem,
    config: Experiment,
    cache_dir: pathlib.Path,
) -> dict[vod_configs.DatasetConfig, vt.Array]:
    module.eval()
    return {
        cfg: compute_vectors(
            dataset=vod_datasets.load_dataset(cfg),  # type: ignore
            module=module,
            fabric=fabric,
            save_dir=cache_dir,
            collate_config=config.collates.predict,
            dataloader_config=config.dataloaders.predict,
            field=cfg.field,
            locator=cfg.identifier,
        )
        for cfg in dset_configs
    }


def _train_for_period(
    *,
    module: vod_models.VodSystem,
    optimizer: torch.optim.Optimizer,
    scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
    fabric: L.Fabric,
    config: Experiment,
    state: TrainerState,
    cache_dir: pathlib.Path,
) -> TrainerState:
    if helpers.is_engine_enabled(state.get_parameters(), "dense"):
        vectors = _compute_all_vectors(
            dset_configs=config.get_dataset_configs(what="all", split="train+val"),
            fabric=fabric,
            module=module,
            config=config,
            cache_dir=cache_dir,
        )
    else:
        vectors = None

    return spawn_search_and_train(
        state=state,
        module=module,
        optimizer=optimizer,
        scheduler=scheduler,
        fabric=fabric,
        train_queries=schemas.QueriesWithVectors.from_configs(
            queries=config.datasets.training.queries.train,
            vectors=vectors,
        ),
        val_queries=schemas.QueriesWithVectors.from_configs(
            queries=config.datasets.training.queries.val,
            vectors=vectors,
        ),
        sections=schemas.SectionsWithVectors.from_configs(
            sections=config.datasets.training.sections.sections,
            vectors=vectors,
        ),
        collate_config=config.collates.train,
        train_dataloader_config=config.dataloaders.train,
        eval_dataloader_config=config.dataloaders.eval,
        cache_dir=cache_dir,
        serve_on_gpu=False,
    )


def _run_benchmarks(
    *,
    module: vod_models.VodSystem,
    fabric: L.Fabric,
    config: Experiment,
    state: TrainerState,
    cache_dir: pathlib.Path,
    barrier: typ.Callable[[str], None],
) -> None:
    if len(config.datasets.benchmark) == 0:
        return

    barrier("Running benchmarks..")
    if helpers.is_engine_enabled(config.benchmark.parameters, "dense"):
        vectors = _compute_all_vectors(
            dset_configs=config.get_dataset_configs(what="all", split="benchmark"),
            fabric=fabric,
            module=module,
            config=config,
            cache_dir=cache_dir,
        )
    else:
        vectors = None

    logger.info(f"Running benchmarks ... (period={1+state.pidx})")
    for j, task in enumerate(config.datasets.benchmark):
        bench_loc = f"{task.queries.identifier} <- {task.sections.identifier}"
        logger.info(f"{1+j}/{len(config.datasets.benchmark)} - Benchmarking `{bench_loc}` ...")

        # Run the benchmark and log the results
        metrics = benchmark_retrieval(
            queries=schemas.QueriesWithVectors.from_configs(
                queries=[task.queries],
                vectors=vectors,
            ),
            sections=schemas.SectionsWithVectors.from_configs(
                sections=[task.sections],
                vectors=vectors,
            ),
            fabric=fabric,
            config=config.benchmark,
            collate_config=config.collates.benchmark,
            dataloader_config=config.dataloaders.benchmark,
            cache_dir=cache_dir,
            device=module.device,
        )
        if metrics is not None:
            metrics: typ.Mapping[str, float | torch.Tensor] = {
                k: fabric.all_reduce(v, reduce_op="mean") for k, v in metrics.items()  # type: ignore
            }
            if fabric.is_global_zero:
                header = f"{bench_loc} - Period {state.pidx + 1}"
                logging.log(
                    {f"{task.queries.identifier}/{k}": v for k, v in metrics.items()},
                    loggers=fabric.loggers,
                    console_header=header,
                    step=state.step,
                    console=True,
                    console_exclude="(?!diagnostics)",
                )

    barrier("Benchmarks completed.")
