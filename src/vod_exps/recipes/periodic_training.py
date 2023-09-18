import functools
import pathlib
import typing as typ

import lightning as L
import numpy as np
import rich
import transformers
import vod_configs
import vod_datasets
import vod_models
import vod_types as vt
from loguru import logger
from vod_dataloaders.dl_sampler import dl_sampler_factory
from vod_tools import cache_manager, pretty
from vod_tools.pretty.print_metrics import pprint_metric_dict
from vod_workflows.evaluation.retrieval import ToDiskConfig, benchmark_retrieval
from vod_workflows.processing.vectors import compute_vectors
from vod_workflows.training.train import spawn_search_and_train
from vod_workflows.utils import helpers, io, logging, schemas


def periodic_training(  # noqa: C901, PLR0915
    *,
    fabric: L.Fabric,
    ranker: vod_models.Ranker,
    config: vod_configs.PeriodicTrainingConfig,
    resume_from_checkpoint: None | str = None,
) -> vod_models.Ranker:
    """Train a ranking model while periodically updating the index."""
    barrier = functools.partial(helpers.barrier_fn, fabric=fabric)

    # Setting up the optimizer & state
    optimizer = ranker.get_optimizer()
    scheduler = ranker.get_scheduler(optimizer)
    state = helpers.TrainerState.from_config(config, fabric)

    # Load an existing checkpoint
    if resume_from_checkpoint is not None:
        logger.info(f"Loading training state from `{config.trainer.checkpoint_path}`")
        io.load_training_state(
            checkpoint_path=resume_from_checkpoint,
            fabric=fabric,
            model=ranker,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=state,
        )
        rich.print(state)

    # Setup Fabric model & optimizer
    ranker, optimizer = fabric.setup(ranker, optimizer)

    # Get the tokenizer
    tokenizer = config.tokenizer.instantiate()

    # Train the model for each period
    barrier("Training starting..")
    for pidx in _iter_periods(state):
        # fetch the parameters for this period
        train_parameters = state.get_parameters()
        bench_parameters = config.benchmark.parameters
        run_benchmarks = pidx > 0 or config.benchmark.on_init

        # Wrap everything in a temporary directory to avoid filling up the disk.
        # The temporary directory will be deleted at the end of each period except the first one
        # as dataset vectors won't change when using the same seed/model.
        with cache_manager.CacheManager(
            pathlib.Path(config.sys.cache_dir, f"train-with-updates-{state.pidx}"),
            delete_existing=False,
            persist=state.pidx == 0,  # keep the cache for the first period
        ) as cache_dir:
            all_dset_configs = list(config.dataset.get_dataset_configs(split="all" if run_benchmarks else "train+val"))
            # pre-process all datasets on rank zero on each node,
            # loading & preprocessing happens implicitely when loading a dataset
            if state.pidx == 0 and fabric.local_rank == 0:
                for cfg in all_dset_configs:
                    vod_datasets.load_dataset(cfg)
            barrier("Pre-processing done.")

            # pre-compute the vectors for each dataset, this is deactivated
            #   when faiss/qdrant is not in use. All vectors are store on disk thanks to `TensorStore`.
            if helpers.is_engine_enabled(train_parameters, "dense") or (
                run_benchmarks and helpers.is_engine_enabled(bench_parameters, "dense")
            ):
                vectors = _compute_all_vectors(
                    dset_configs=all_dset_configs,
                    fabric=fabric,
                    ranker=ranker,
                    config=config,
                    tokenizer=tokenizer,
                    cache_dir=cache_dir,
                )
            else:
                logger.info("Dense search engine is disabled. Skipping vector pre-computation.")
                vectors = None

            # Run the benchmarks before each training period
            if run_benchmarks:
                barrier("Running benchmarks..")
                _run_benchmarks(
                    fabric=fabric,
                    vectors=vectors,
                    config=config,
                    state=state,
                    parameters=bench_parameters,
                    cache_dir=cache_dir,
                    tokenizer=tokenizer,
                )
                barrier("Completed benchmarks.")
                raise NotImplementedError()

            if state.period_max_steps is None:
                # If there is no defined `end_step`, we reached the end of the training
                continue

            # Training the model for the current period.
            # We use a `StopAtTrainer` to stop the training at the end of the current period (max `end_step`).
            logger.info(f"Starting training period {1+state.pidx}")
            state = spawn_search_and_train(
                ranker=ranker,
                optimizer=optimizer,
                scheduler=scheduler,
                fabric=fabric,
                train_queries=schemas.QueriesWithVectors.from_configs(
                    queries=config.dataset.training.queries.train,
                    vectors=vectors,
                ),
                val_queries=schemas.QueriesWithVectors.from_configs(
                    queries=config.dataset.training.queries.val,
                    vectors=vectors,
                ),
                sections=schemas.SectionsWithVectors.from_configs(
                    sections=config.dataset.training.sections.sections,
                    vectors=vectors,
                ),
                tokenizer=tokenizer,
                collate_config=config.collates.train,
                train_dataloader_config=config.dataloaders.train,
                eval_dataloader_config=config.dataloaders.eval,
                dl_sampler=dl_sampler_factory(config.dl_sampler) if config.dl_sampler is not None else None,
                cache_dir=cache_dir,
                serve_on_gpu=False,
                trainer_state=state,
                checkpoint_path=config.trainer.checkpoint_path,
                on_first_batch_fn=functools.partial(
                    _on_first_batch_fn,
                    tokenizer=tokenizer,
                    output_file=pathlib.Path(f"{state.step}-training-batch.txt"),
                ),
                pbar_keys=config.trainer.pbar_keys,
            )
            logger.success(f"Completed training period {1+state.pidx}")
            barrier("Period completed.")

        # Reset the scheduler for the next period
        scheduler = ranker.get_scheduler(helpers.unwrap_optimizer(optimizer))

    return ranker


def _compute_all_vectors(
    *,
    dset_configs: list[vod_configs.DatasetConfig],
    fabric: L.Fabric,
    ranker: vod_models.Ranker,
    config: vod_configs.PeriodicTrainingConfig,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    cache_dir: pathlib.Path,
) -> dict[vod_configs.DatasetConfig, vt.Array]:
    return {
        cfg: compute_vectors(
            dataset=vod_datasets.load_dataset(cfg),  # type: ignore
            ranker=ranker,
            tokenizer=tokenizer,
            fabric=fabric,
            cache_dir=cache_dir,
            collate_config=config.collates.predict,
            dataloader_config=config.dataloaders.predict,
            field=cfg.field,
            locator=cfg.identifier,
        )
        for cfg in dset_configs
    }


def _on_first_batch_fn(
    fabric: L.Fabric,
    batch: dict[str, typ.Any],
    ranker: vod_models.Ranker,  # noqa: ARG001
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
    output_file: None | pathlib.Path = None,
) -> None:
    if fabric.is_global_zero:
        pretty.pprint_batch(batch, header="Training batch")
        pretty.pprint_retrieval_batch(batch, tokenizer=tokenizer, skip_special_tokens=True, output_file=output_file)
        if output_file is not None:
            try:
                import wandb

                # log th output file to wandb
                wandb.log({"data/retrieval-batch": wandb.Html(open(output_file).read())})  # noqa: SIM115
            except Exception as exc:
                logger.debug(f"Failed to log to wandb: {exc}")


def _run_benchmarks(
    *,
    fabric: L.Fabric,
    config: vod_configs.PeriodicTrainingConfig,
    vectors: None | dict[vod_configs.DatasetConfig, vt.Array],
    state: helpers.TrainerState,
    parameters: dict[str, float],
    cache_dir: pathlib.Path,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
) -> None:
    # if helpers.is_engine_enabled(parameters, "faiss"):
    # ranker.to("cpu")  # <-- free GPU memory # see: https://github.com/Lightning-AI/lightning/issues/17937
    if fabric.is_global_zero:
        logger.info(f"Running benchmarks ... (period={1+state.pidx})")
        for j, task in enumerate(config.dataset.benchmark):
            bench_loc = f"{task.queries.identifier} <- {task.queries.identifier}"
            logger.info(f"{1+j}/{len(config.dataset.benchmark)} - Benchmarking `{bench_loc}` ...")
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
                tokenizer=tokenizer,
                metrics=config.benchmark.metrics,
                collate_config=config.collates.benchmark,
                dataloader_config=_patch_mum_worker(
                    config.dataloaders.benchmark,
                    fabric=fabric,
                ),
                cache_dir=cache_dir,
                parameters=parameters,
                serve_on_gpu=True,
                n_max=config.benchmark.n_max_eval,
                to_disk_config=ToDiskConfig(
                    logdir=logdir,
                    tokenizer=config.tokenizer.instantiate(),
                ),
            )
            if metrics is not None:
                header = f"{bench_loc} - Period {state.pidx + 1}"
                _log_print_metrics(fabric=fabric, state=state, locator=bench_loc, metrics=metrics, header=header)
            logger.info(f"{1+j}/{len(config.dataset.benchmark)} - saved to `{logdir.absolute()}`")


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
    state: helpers.TrainerState,
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


def _iter_periods(trainer_state: helpers.TrainerState) -> typ.Iterable[int]:
    update_steps = _infer_update_steps(trainer_state.max_steps, trainer_state.period)
    logger.info(f"The search index will be updated at steps: {_pretty_steps(update_steps)}")
    if len(update_steps) == 0:
        raise ValueError("No index update steps were defined.")

    for pidx, (start_step, end_step) in enumerate(zip(update_steps, update_steps[1:] + [None])):
        if end_step is not None and trainer_state.step > end_step:  # type: ignore
            logger.info(f"Skipping period {pidx + 1}/{len(update_steps)} (step {start_step} -> {end_step})")
            continue

        # Update the Trainer state
        trainer_state.pidx = pidx
        trainer_state.period_max_steps = end_step  # type: ignore
        yield pidx


def _infer_update_steps(total_number_of_steps: int, update_freq: int | list[int]) -> list[int]:
    if isinstance(update_freq, int):
        steps = [int(x) for x in np.arange(0, total_number_of_steps, update_freq)]
    elif isinstance(update_freq, list):
        if update_freq[0] != 0:
            update_freq = [0] + update_freq
        if update_freq[-1] == total_number_of_steps:
            update_freq = update_freq[:-1]
        steps = update_freq
    else:
        raise TypeError(f"Invalid type for `update_freq`: {type(update_freq)}")

    return steps + [total_number_of_steps]


def _pretty_steps(steps: list[int], max_steps: int = 6) -> str:
    steps = steps[:-1]
    if len(steps) > max_steps:
        return f"[{steps[0]}, {steps[1]}, {steps[2]}, {steps[3]}, {steps[4]} ... {steps[-1]}]"

    return str(steps)
