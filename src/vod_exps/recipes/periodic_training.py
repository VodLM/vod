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
from vod_tools import cache_manager
from vod_tools.pretty.print_metrics import pprint_metric_dict
from vod_workflows.evaluation.retrieval import ToDiskConfig, benchmark_retrieval
from vod_workflows.processing.vectors import compute_vectors
from vod_workflows.training.train import spawn_search_and_train
from vod_workflows.utils import TrainerState, helpers, io, logging, schemas


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
            cache_dir=cache_dir,
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
        checkpoint_path=config.trainer.checkpoint_path,
        pbar_keys=config.trainer.pbar_keys,
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

    if fabric.is_global_zero:
        logger.info(f"Running benchmarks ... (period={1+state.pidx})")
        for j, task in enumerate(config.datasets.benchmark):
            bench_loc = f"{task.queries.identifier} <- {task.queries.identifier}"
            logger.info(f"{1+j}/{len(config.datasets.benchmark)} - Benchmarking `{bench_loc}` ...")
            logdir = pathlib.Path("benchmarks", f"{bench_loc}-{state.pidx}-{state.step}")
            logdir.mkdir(parents=True, exist_ok=True)

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
                metrics=config.benchmark.metrics,
                collate_config=config.collates.benchmark,
                dataloader_config=_patch_mum_worker(
                    config.dataloaders.benchmark,
                    fabric=fabric,
                ),
                cache_dir=cache_dir,
                parameters=config.benchmark.parameters,
                serve_search_on_gpu=True,
                n_max=config.benchmark.n_max_eval,
                to_disk_config=ToDiskConfig(
                    logdir=logdir,
                    tokenizer=config.collates.benchmark.tokenizer_encoder.instantiate(),
                ),
            )
            if metrics is not None:
                header = f"{bench_loc} - Period {state.pidx + 1}"
                _log_print_metrics(
                    fabric=fabric, state=state, locator=task.queries.identifier, metrics=metrics, header=header
                )
            logger.info(f"{1+j}/{len(config.datasets.benchmark)} - saved to `{logdir.absolute()}`")

    barrier("Benchmarks completed.")


def _patch_mum_worker(
    dl_config: vod_configs.DataLoaderConfig,
    fabric: L.Fabric,
    overrides: None | dict[str, typ.Any] = None,
) -> vod_configs.DataLoaderConfig:
    return dl_config.model_copy(
        update={
            "num_workers": fabric.world_size * dl_config.num_workers,
            **(overrides or {}),
        }
    )


def _log_print_metrics(
    *,
    fabric: L.Fabric,
    state: TrainerState,
    locator: str,
    metrics: dict[str, float],
    header: str,
) -> None:
    locator = locator.replace(":", "/")
    logging.log(
        {f"{locator}/{k}": v for k, v in metrics.items()},
        loggers=fabric.loggers,
        header=header,
        step=state.step,
    )
    pprint_metric_dict(
        {f"{locator}/{k}": v for k, v in metrics.items() if "diagnostics" not in k},
        header=header,
    )
