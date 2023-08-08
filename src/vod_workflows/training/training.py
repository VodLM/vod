from __future__ import annotations

import collections
import dataclasses
import functools
import multiprocessing as mp
import pathlib
import time
import typing
from multiprocessing.managers import DictProxy

import lightning as L
import numpy as np
import rich
import torch
import transformers
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger
from rich import progress
from torch.utils import data as torch_data
from vod_tools import dstruct, pipes
from vod_tools.misc.progress import IterProgressBar
from vod_workflows.utils import helpers, io

from src import vod_configs, vod_dataloaders, vod_datasets, vod_models, vod_search

K = typing.TypeVar("K")


class OnFirstBatchCallback(typing.Protocol):
    """A callback that is called on the first batch of the first epoch."""

    def __call__(self, fabric: L.Fabric, batch: dict[str, torch.Tensor], ranker: vod_models.Ranker) -> None:
        """Do some stuff."""
        ...


@dataclasses.dataclass(frozen=True)
class RetrievalTask:
    """Holds the train and validation datasets."""

    train_questions: helpers.DsetWithVectors
    val_questions: helpers.DsetWithVectors
    sections: helpers.DsetWithVectors


def index_and_train(
    *,
    ranker: vod_models.Ranker,
    optimizer: torch.optim.Optimizer,
    scheduler: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    trainer_state: helpers.TrainerState,
    fabric: L.Fabric,
    train_factories: dict[K, vod_datasets.RetrievalDatasetFactory],
    val_factories: dict[K, vod_datasets.RetrievalDatasetFactory],
    vectors: dict[K, None | helpers.PrecomputedDsetVectors],
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    search_config: vod_configs.SearchConfig,
    collate_config: vod_configs.RetrievalCollateConfig,
    train_dataloader_config: vod_configs.DataLoaderConfig,
    eval_dataloader_config: vod_configs.DataLoaderConfig,
    dl_sampler: typing.Optional[vod_dataloaders.SamplerFactory] = None,
    cache_dir: pathlib.Path,
    serve_on_gpu: bool = False,
    checkpoint_path: typing.Optional[str] = None,
    on_first_batch_fn: typing.Optional[OnFirstBatchCallback] = None,
    pbar_keys: typing.Optional[list[str]] = None,
) -> helpers.TrainerState:
    """Index the sections and train the ranker."""
    barrier_fn = functools.partial(helpers.barrier_fn, fabric=fabric)

    # Gather datasets and their corresponding vectors
    task = _make_retrieval_task(
        train_factories=train_factories,
        val_factories=val_factories,
        vectors=vectors,
    )
    if fabric.is_global_zero:
        rich.print(task)

    # parameters
    parameters = mp.Manager().dict()
    parameters.update(trainer_state.get_parameters())

    # free GPU resources see: https://github.com/Lightning-AI/lightning/issues/17937
    # if helpers.is_engine_enabled(parameters, "faiss"):
    #     ranker.to("cpu")

    barrier_fn("Init search engines..")
    with vod_search.build_multi_search_engine(
        sections=task.sections.data,
        vectors=task.sections.vectors,
        config=search_config,
        cache_dir=cache_dir,
        dense_enabled=helpers.is_engine_enabled(parameters, "faiss"),
        sparse_enabled=True,
        skip_setup=not fabric.is_global_zero,
        barrier_fn=barrier_fn,
        serve_on_gpu=serve_on_gpu,
    ) as master:
        barrier_fn("Initiating dataloaders")
        search_client = master.get_client()

        # instantiate the dataloader
        logger.debug("Instantiating dataloader..")
        init_dataloader = functools.partial(
            helpers.instantiate_retrieval_dataloader,
            sections=task.sections,
            tokenizer=tokenizer,
            search_client=search_client,
            collate_config=collate_config,
            parameters=parameters,
            dl_sampler=dl_sampler,
        )

        # patching dataloaders with `Fabric` methods
        train_dl, val_dl = fabric.setup_dataloaders(
            *(
                init_dataloader(**cfg)  # type: ignore
                for cfg in (
                    {"questions": task.train_questions, "dataloader_config": train_dataloader_config},
                    {"questions": task.val_questions, "dataloader_config": eval_dataloader_config},
                )
            ),
            use_distributed_sampler=dl_sampler is None,
            move_to_device=True,
        )

        # train the ranker
        barrier_fn(f"Starting training period {1+trainer_state.period}")
        trainer_state = _training_loop(
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
        barrier_fn(f"Completed period {1+trainer_state.period}")

    return trainer_state


def _training_loop(  # noqa: C901, PLR0915
    *,
    ranker: vod_models.Ranker,
    optimizer: torch.optim.Optimizer,
    trainer_state: helpers.TrainerState,
    fabric: L.Fabric,
    train_dl: torch_data.DataLoader,
    val_dl: torch_data.DataLoader,
    scheduler: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_path: None | str = None,
    on_first_batch_fn: None | OnFirstBatchCallback = None,
    pbar_keys: None | typing.List[str] = None,
    parameters: None | DictProxy = None,
) -> helpers.TrainerState:
    _check_ranker_and_optimizer(ranker, optimizer)
    optimizer.zero_grad()
    ranker.train()

    if fabric.is_global_zero:
        rich.print(trainer_state)

    try:
        # infer the number of training and valid steps
        n_train_steps, n_val_steps = _infer_num_steps(trainer_state, fabric, val_dl)
        with IterProgressBar(disable=not fabric.is_global_zero) as pbar:
            train_pbar = pbar.add_task(
                f"Period {1+trainer_state.period}",
                total=n_train_steps,
                info=_pbar_info(trainer_state, keys=pbar_keys),
            )
            eval_metrics = None
            chrono = None
            local_step = 0
            max_steps = trainer_state.period_max_steps or trainer_state.max_steps
            while trainer_state.step < max_steps:
                for batch in train_dl:
                    if trainer_state.step >= max_steps:
                        break

                    if on_first_batch_fn is not None and local_step == 0:
                        on_first_batch_fn(fabric, batch, ranker)

                    # Forward pass
                    is_accumulating = (1 + local_step) % trainer_state.accumulate_grad_batches != 0
                    step_metrics = ranker.gradients.forward_backward(
                        batch=batch,
                        fwd_fn=ranker,
                        fabric=fabric,
                        loss_scaler=1 / trainer_state.accumulate_grad_batches,
                        no_backward_sync=is_accumulating,
                    )

                    # Log the training metrics
                    if trainer_state.step % trainer_state.log_interval == 0:
                        fabric.log_dict(
                            metrics={
                                "trainer/epoch": float(trainer_state.epoch),
                                **{f"train/{k.replace('.', '/')}": v for k, v in step_metrics.items()},
                                **{f"parameter/{k}": v for k, v in _extract_learning_rates(optimizer).items()},
                            },
                            step=trainer_state.step,
                        )

                    # Optimization & eval step
                    if not is_accumulating:
                        # Clip the gradients
                        if trainer_state.gradient_clip_val is not None:
                            fabric.clip_gradients(ranker, optimizer, max_norm=trainer_state.gradient_clip_val)

                        # Run an optimization step, reset the gradients and update the learning rate
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

                        # Update the chrono, the trainer state and the progress bar
                        if chrono is not None:
                            chrono.stop()
                        trainer_state.step += 1

                        # Update the parameters
                        if parameters is not None:
                            parameters.update(trainer_state.get_parameters())

                        # Update the progress bar
                        pbar.update(
                            train_pbar,
                            advance=1,
                            speed=chrono.get_avg_laps_per_second() if chrono is not None else None,
                            info=_pbar_info(
                                trainer_state,
                                train_metrics=step_metrics,
                                eval_metrics=eval_metrics,
                                keys=pbar_keys,
                            ),
                        )

                        # Validation
                        if (1 + trainer_state.step) % trainer_state.val_check_interval == 0:
                            optimizer.zero_grad()
                            eval_metrics = _validation_loop(
                                ranker=ranker,
                                fabric=fabric,
                                trainer_state=trainer_state,
                                val_dl=val_dl,
                                n_steps=n_val_steps,
                                pbar=pbar,
                            )
                            if checkpoint_path is not None:
                                io.save_training_state(
                                    checkpoint_path=checkpoint_path,
                                    fabric=fabric,
                                    model=ranker,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    trainer_state=trainer_state,
                                )

                        # start the chono
                        if chrono is None:
                            chrono = Chrono()
                        chrono.start()

                    local_step += 1
                trainer_state.epoch += 1
    except KeyboardInterrupt:
        logger.warning(
            f"Training period {1+trainer_state.period} (step={trainer_state.step}) "
            f"interrupted by user (KeyboardInterrupt)."
        )

    optimizer.zero_grad()

    # Save and Synch the model parameters
    if checkpoint_path is not None:
        logger.info("End of period. Saving and re-loading model from checkpoint (parameter sync).")
        io.save_training_state(
            checkpoint_path=checkpoint_path,
            fabric=fabric,
            model=ranker,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )
        io.load_training_state(
            checkpoint_path=checkpoint_path,
            fabric=fabric,
            model=ranker,
        )
    logger.info(f"End of period. Model hash: `{pipes.fingerprint_torch_module(None, ranker)}`")
    return trainer_state


def _check_ranker_and_optimizer(ranker: vod_models.Ranker, optimizer: torch.optim.Optimizer) -> None:
    if not fabric_wrappers.is_wrapped(ranker):
        raise RuntimeError("Ranker must be wrapped by lightning `Fabric`.")
    if not fabric_wrappers.is_wrapped(optimizer):
        raise RuntimeError("Optimizer must be wrapped by lightning `Fabric`.")


def _infer_num_steps(
    trainer_state: helpers.TrainerState,
    fabric: L.Fabric,
    val_dl: torch_data.DataLoader,
) -> tuple[int, int]:
    max_steps = trainer_state.period_max_steps or trainer_state.max_steps
    n_train_steps = max_steps - trainer_state.step
    if trainer_state.n_max_eval is None:
        n_val_steps = len(val_dl)
    else:
        eff_eval_bs = fabric.world_size * val_dl.batch_size  # type: ignore
        n_val_steps = min(len(val_dl), max(1, -(-trainer_state.n_max_eval // eff_eval_bs)))
    return n_train_steps, n_val_steps


@torch.no_grad()
def _validation_loop(
    ranker: vod_models.Ranker,
    fabric: L.Fabric,
    trainer_state: helpers.TrainerState,
    val_dl: torch_data.DataLoader,
    n_steps: int,
    pbar: progress.Progress,
) -> dict[str, float | torch.Tensor]:
    ranker.eval()
    val_pbar = pbar.add_task("Validation", total=n_steps, info=f"{n_steps} steps")
    metrics = collections.defaultdict(list)
    for i, batch in enumerate(val_dl):
        output = ranker(batch, mode="evaluate")
        pbar.update(val_pbar, advance=1, taskinfo=_pbar_info(trainer_state, output))
        for k, v in output.items():
            metrics[k].append(_format_metric(v))
        if i >= n_steps:
            break

    # aggregate metrics
    metrics = {k: np.mean(v) for k, v in metrics.items()}

    # log metrics
    fabric.log_dict(
        metrics={f"val/{k.replace('.', '/')}": v for k, v in metrics.items()},
        step=trainer_state.step,
    )

    pbar.remove_task(val_pbar)
    ranker.train()
    return dict(metrics)


def _format_metric(v: typing.Any) -> typing.Any:  # noqa: ANN401
    if isinstance(v, torch.Tensor):
        return v.detach().mean().cpu()

    return v


class Chrono:
    """A simple chronometer."""

    _laps: list[tuple[float, float]]
    _start_time: typing.Optional[float]

    def __init__(self, buffer_size: int = 100) -> None:
        self._laps = []
        self._start_time = None
        self.buffer_size = buffer_size

    def reset(self) -> "Chrono":
        """Reset the chrono."""
        self._laps = []
        self._start_time = None
        return self

    def start(self) -> "Chrono":
        """Start the chrono."""
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> "Chrono":
        """Stop the chrono."""
        if self._start_time is None:
            raise RuntimeError("Chrono is not running")
        curr_time = time.perf_counter()
        self._laps.append((self._start_time, curr_time))
        if len(self._laps) > self.buffer_size:
            self._laps.pop(0)
        self._start_time = None
        return self

    def get_total_time(self) -> float:
        """Return the total time elapsed since the chrono was started."""
        return sum(end - start for start, end in self._laps)

    def get_avg_time(self) -> float:
        """Return the average time elapsed since the chrono was started."""
        return self.get_total_time() / len(self._laps)

    def get_avg_laps_per_second(self) -> float:
        """Return the average number of laps per second."""
        return len(self._laps) / self.get_total_time()


def _extract_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {f"lr_{i}": param_group["lr"] for i, param_group in enumerate(optimizer.param_groups)}


def _pbar_info(
    state: helpers.TrainerState,
    train_metrics: typing.Optional[dict[str, typing.Any]] = None,
    eval_metrics: typing.Optional[dict[str, typing.Any]] = None,
    keys: typing.Optional[list[str]] = None,
) -> str:
    keys = keys or ["loss"]
    desc = (
        f"{1+state.step}/{state.period_max_steps} ({state.max_steps}) "
        f"• epoch={1+state.epoch} "
        f"• grad-acc={state.accumulate_grad_batches}"
    )
    if train_metrics or eval_metrics:
        suppl = []
        if train_metrics is not None:
            for k in keys:
                if k in train_metrics:
                    suppl.append(f"train/{k}={train_metrics[k]:.3f}")

        if eval_metrics is not None:
            for k in keys:
                if k in eval_metrics:
                    suppl.append(f"val/{k}={eval_metrics[k]:.3f}")

        desc = f"[yellow]{' '.join(suppl)}[/yellow] • {desc}"

    return desc


def _make_retrieval_task(
    train_factories: dict[K, vod_datasets.RetrievalDatasetFactory],
    val_factories: dict[K, vod_datasets.RetrievalDatasetFactory],
    vectors: dict[K, helpers.PrecomputedDsetVectors | None],
) -> RetrievalTask:
    """Create the `RetrievalTask` from the training and validation factories."""

    def _vec(key: K, field: typing.Literal["question", "section"]) -> None | dstruct.TensorStoreFactory:
        """Safely fetch the relevant `vector` from the `PrecomputedDsetVectors` structure."""
        x = vectors[key]
        if x is None:
            return None
        if field == "question":
            return x.questions
        if field == "section":
            return x.sections
        raise ValueError(f"Unknown field: {field}")

    return RetrievalTask(
        train_questions=helpers.concatenate_datasets(
            [
                helpers.DsetWithVectors.cast(data=factory.get_queries(), vectors=_vec(key, "question"))
                for key, factory in train_factories.items()
            ]
        ),
        val_questions=helpers.concatenate_datasets(
            [
                helpers.DsetWithVectors.cast(data=factory.get_queries(), vectors=_vec(key, "question"))
                for key, factory in val_factories.items()
            ]
        ),
        sections=helpers.concatenate_datasets(
            [
                helpers.DsetWithVectors.cast(data=factory.get_sections(), vectors=_vec(key, "section"))
                for key, factory in {**train_factories, **val_factories}.items()
            ]
        ),
    )
