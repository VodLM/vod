from __future__ import annotations

import functools
import pathlib
from typing import Any, Iterable, Optional, TypeVar

import lightning as L
import numpy as np
import omegaconf
import rich
import transformers
from loguru import logger
from vod_tools import cache_manager, pipes
from vod_tools.misc.pretty import print_metric_groups
from vod_workflows.evaluation.evaluation import ToDiskConfig, benchmark
from vod_workflows.support.precompute import compute_vectors
from vod_workflows.support.tuning import tune_parameters
from vod_workflows.training.training import index_and_train
from vod_workflows.utils import helpers, io, logging

from src import vod_configs, vod_datasets, vod_models

K = TypeVar("K")


def train_with_index_updates(  # noqa: C901, PLR0915
    *,
    fabric: L.Fabric,
    ranker: vod_models.Ranker,
    config: vod_configs.TrainWithIndexUpdatesConfigs | omegaconf.DictConfig,
    load_from: Optional[str] = None,  # todo
) -> vod_models.Ranker:
    """Train a ranking model while periodically updating the index."""
    barrier = functools.partial(helpers.barrier_fn, fabric=fabric)
    if isinstance(config, omegaconf.DictConfig):
        config = vod_configs.TrainWithIndexUpdatesConfigs.parse(config)

    # Define the index update steps and the `PeriodicStoppingCallback` callback.
    update_steps = _infer_update_steps(config.trainer.max_steps, config.trainer.period)
    logger.info(f"The search index will be updated at steps: {_pretty_steps(update_steps)}")
    if len(update_steps) == 0:
        raise ValueError("No index update steps were defined.")

    # Setting up the optimizer
    optimizer = ranker.get_optimizer()
    scheduler = ranker.get_scheduler(helpers.unwrap_optimizer(optimizer))

    # Define the trainer State
    state = helpers.TrainerState(
        step=0,
        period=0,
        epoch=0,
        period_max_steps=None,
        max_steps=config.trainer.max_steps,
        log_interval=config.trainer.log_interval,
        val_check_interval=config.trainer.val_check_interval,
        n_max_eval=config.trainer.n_max_eval,
        accumulate_grad_batches=_infer_accumulate_grad_batches(fabric, config.batch_size),
        gradient_clip_val=config.trainer.gradient_clip_val,
        parameters=config.trainer.parameters,
    )

    if load_from is not None:
        logger.info(f"Loading training state from `{load_from}`")
        io.load_training_state(
            checkpoint_path=load_from,
            fabric=fabric,
            model=ranker,
            optimizer=optimizer,
            scheduler=None,
            trainer_state=state,
        )
        rich.print(state)

    # Setup Fabric model & optimizer
    ranker, optimizer = fabric.setup(ranker, optimizer)

    # Train the model for each period
    barrier("Training starting..")
    for pidx, (start_step, end_step) in enumerate(zip(update_steps, update_steps[1:] + [None])):
        if end_step is not None and state.step > end_step:
            logger.info(f"Skipping period {pidx + 1}/{len(update_steps)} (step {start_step} -> {end_step})")
            continue

        # Update the Trainer state
        state.period = pidx
        state.period_max_steps = end_step

        # fetch the parameters for this period
        train_parameters = state.get_parameters()
        bench_parameters = config.benchmark.parameters
        run_benchmarks = state.period > 0 or config.benchmark.on_init

        # wrap everything in a temporary directory to avoid filling up the disk.
        # The temporary directory will be deleted at the end of each period except the first one
        # as dataset vectors won't change when using the same seed/model.
        with cache_manager.CacheManager(
            pathlib.Path(config.sys.cache_dir, f"train-with-updates-{state.period}"),
            delete_existing=False,
            persist=state.period == 0,  # keep the cache for the first period
        ) as cache_dir:
            factories = _get_dset_factories(
                config.dataset.get("all" if run_benchmarks else "train+val"),
                config=config.dataset,
            )
            # pre-process all datasets on global zero, this is done implicitely when loading a dataset
            if pidx == 0:
                if fabric.is_global_zero:
                    for key, factory in factories.items():
                        logger.debug(f"Pre-processing `{key}` ...")
                        factory.get_qa_split()
                        factory.get_sections()
                barrier("Pre-processing done.")

            # pre-compute the vectors for each dataset, this is deactivated when faiss is not in use
            if helpers.is_engine_enabled(train_parameters, "faiss") or (
                run_benchmarks and helpers.is_engine_enabled(bench_parameters, "faiss")
            ):
                # Compute the vectors
                vectors = compute_vectors(
                    factories,
                    ranker=ranker,
                    tokenizer=config.tokenizer.instantiate(),
                    fabric=fabric,
                    cache_dir=cache_dir,
                    collate_config=config.collates.predict,
                    dataloader_config=config.dataloaders.predict,
                )
            else:
                logger.info("Faiss engine is disabled. Skipping vector pre-computation.")
                vectors = {dset: None for dset in factories}

            # Tune the parameters and benchmark the ranker on each dataset separately
            if run_benchmarks:
                if config.benchmark.tuning is not None and bench_parameters.get("faiss", -1) > 0:
                    barrier("Tuning retrieval parameters..")
                    bench_parameters = tune_parameters(
                        parameters=bench_parameters,
                        tune=["sparse"],
                        fabric=fabric,
                        factories=_get_dset_factories(config.dataset.get("val"), config=config.dataset),
                        vectors=vectors,
                        search_config=config.search_defaults,
                        collate_config=config.collates.train.copy(update=config.benchmark.tuning.collate_overrides),
                        dataloader_config=_patch_mum_worker(
                            config.dataloaders.benchmark,
                            fabric=fabric,
                            overrides={"batch_size": config.benchmark.tuning.batch_size},
                        ),
                        cache_dir=cache_dir,
                        serve_on_gpu=True,
                        tuning_steps=config.benchmark.tuning.steps,
                        learning_rate=config.benchmark.tuning.learning_rate,
                        tokenizer=config.tokenizer.instantiate(),
                    )
                    logger.info(f"Tuned parameters: {bench_parameters}")

                barrier("Running benchmarks..")
                _run_benchmarks(
                    fabric=fabric,
                    config=config,
                    state=state,
                    parameters=bench_parameters,
                    cache_dir=cache_dir,
                    factories=factories,
                    vectors=vectors,
                )
                barrier("Completed benchmarks.")

            if end_step is None:
                # If there is no defined `end_step`, we reached the end of the training
                continue

            # training for the current period.
            # We use a `StopAtTrainer` to stop the training at the end of the current period (max `end_step`).
            logger.info(f"Starting training period {1+state.period}")
            state = index_and_train(
                ranker=ranker,
                optimizer=optimizer,
                scheduler=scheduler,
                fabric=fabric,
                vectors=vectors,  # type: ignore
                train_factories=_get_dset_factories(config.dataset.get("train"), config=config.dataset),
                val_factories=_get_dset_factories(config.dataset.get("val"), config=config.dataset),
                tokenizer=config.tokenizer.instantiate(),
                search_config=config.search_defaults,
                collate_config=config.collates.train,
                train_dataloader_config=config.dataloaders.train,
                eval_dataloader_config=config.dataloaders.eval,
                dl_sampler=config.dl_sampler,
                cache_dir=cache_dir,
                serve_on_gpu=False,
                trainer_state=state,
                checkpoint_path=config.trainer.checkpoint_path,
                on_first_batch_fn=functools.partial(
                    _on_first_batch_fn,
                    tokenizer=config.tokenizer.instantiate(),
                    output_file=pathlib.Path(f"{state.step}-training-batch.txt"),
                ),
                pbar_keys=config.trainer.pbar_keys,
            )
            logger.success(f"Completed training period {1+state.period}")
            barrier("Period completed.")

        # Reset the scheduler for the next period
        scheduler = ranker.get_scheduler(helpers.unwrap_optimizer(optimizer))

    return ranker


def _on_first_batch_fn(
    fabric: L.Fabric,
    batch: dict[str, Any],
    ranker: vod_models.Ranker,  # noqa: ARG001
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
    output_file: Optional[pathlib.Path] = None,
) -> None:
    if fabric.is_global_zero:
        pipes.pprint_batch(batch, header="Training batch")
        pipes.pprint_retrieval_batch(batch, tokenizer=tokenizer, skip_special_tokens=True, output_file=output_file)
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
    config: vod_configs.TrainWithIndexUpdatesConfigs,
    state: helpers.TrainerState,
    parameters: dict[str, float],
    cache_dir: pathlib.Path,
    factories: dict[K, vod_datasets.DatasetFactory],
    vectors: dict[K, None | helpers.PrecomputedDsetVectors],
) -> None:
    # if helpers.is_engine_enabled(parameters, "faiss"):
    # ranker.to("cpu")  # <-- free GPU memory # see: https://github.com/Lightning-AI/lightning/issues/17937
    if fabric.is_global_zero:
        logger.info(f"Running benchmarks ... (period={1+state.period})")
        for j, dset in enumerate(config.dataset.benchmark):
            bench_loc = f"{dset.name}:{dset.split_alias}"
            logger.info(f"{1+j}/{len(config.dataset.benchmark)} - Benchmarking `{bench_loc}` ...")
            logdir = pathlib.Path("benchmarks", f"{bench_loc}-{state.period}-{state.step}")
            logdir.mkdir(parents=True, exist_ok=True)

            # Create the search config
            search_config = config.search_defaults.copy(update=config.benchmark.search)

            # Run the benchmark and log the results
            metrics = benchmark(
                factory=factories[dset],
                vectors=vectors[dset],
                tokenizer=config.tokenizer.instantiate(),
                metrics=config.benchmark.metrics,
                search_config=search_config,
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
                header = f"{dset.name}:{dset.split_alias} - Period {state.period + 1}"
                _log_print_metrics(fabric=fabric, state=state, dset=dset, metrics=metrics, header=header)
            logger.info(f"{1+j}/{len(config.dataset.benchmark)} - saved to `{logdir.absolute()}`")


def _patch_mum_worker(
    dl_config: vod_configs.DataLoaderConfig,
    fabric: L.Fabric,
    overrides: Optional[dict[str, Any]] = None,
) -> vod_configs.DataLoaderConfig:
    return dl_config.copy(
        update={
            "num_workers": fabric.world_size * dl_config.num_workers,
            **(overrides or {}),
        }
    )


def _log_print_metrics(
    *,
    fabric: L.Fabric,
    state: helpers.TrainerState,
    dset: vod_configs.NamedDset,
    metrics: dict[str, float],
    header: str,
) -> None:
    logging.log(
        {f"{dset.name}/{dset.split_alias}/{k}": v for k, v in metrics.items()},
        loggers=fabric.loggers,
        header=header,
        step=state.step,
    )
    print_metric_groups(
        {f"{dset.name}/{dset.split_alias}/{k}": v for k, v in metrics.items() if "diagnostics" not in k},
        header=header,
    )


def _infer_accumulate_grad_batches(fabric: L.Fabric, config: vod_configs.BatchSizeConfig) -> int:
    step_batch_size = fabric.world_size * config.per_device

    # warn if the batch size per step is larger than the effective batch size
    if step_batch_size > config.effective:
        logger.warning(
            f"Effective batch size ({config.effective}) is smaller than the batch size per step "
            f"({step_batch_size}). This will lead to a slower training."
        )
        return 1

    # accumulate gradients if the effective batch size is larger than the batch size per step
    accumulation_steps = -(-config.effective // step_batch_size)

    # warn if the effective batch size is not divisible by the batch size per step
    if config.effective % step_batch_size != 0:
        logger.warning(
            f"The effective batch size ({config.effective}) is not divisible by the batch size per step "
            f"({step_batch_size}). This will lead to a slower training."
        )

    logger.info(
        f"Using {accumulation_steps} accumulation steps. "
        f"Effective batch size: {fabric.world_size * accumulation_steps * config.per_device} "
        f"(requested={config.effective})."
    )
    return accumulation_steps


def _get_dset_factories(
    dsets: Iterable[vod_configs.NamedDset],
    config: vod_configs.MultiDatasetFactoryConfig,
) -> dict[vod_configs.NamedDset, vod_datasets.DatasetFactory]:
    return {
        dset_cfg: vod_datasets.DatasetFactory.from_config(
            config.dataset_factory_config(dset_cfg),
        )
        for dset_cfg in dsets
    }


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
