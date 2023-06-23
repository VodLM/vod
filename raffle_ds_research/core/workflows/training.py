from __future__ import annotations

import collections
import dataclasses
import functools
import pathlib
import time
import typing

import lightning as L
import numpy as np
import rich
import torch
import transformers
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger
from rich import progress
from torch.utils import data as torch_data

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core import mechanics
from raffle_ds_research.core.mechanics import search_engine
from raffle_ds_research.core.mechanics.dataloader_sampler import DataloaderSampler
from raffle_ds_research.core.ml.ranker import Ranker
from raffle_ds_research.core.workflows.precompute import PrecomputedDsetVectors
from raffle_ds_research.core.workflows.utils import io, support
from raffle_ds_research.tools import dstruct
from raffle_ds_research.tools.utils.progress import BatchProgressBar

K = typing.TypeVar("K")


@dataclasses.dataclass(frozen=True)
class RetrievalTask:
    """Holds the train and validation datasets."""

    train_questions: support.DsetWithVectors
    val_questions: support.DsetWithVectors
    sections: support.DsetWithVectors


def index_and_train(
    *,
    ranker: Ranker,
    optimizer: torch.optim.Optimizer,
    scheduler: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    trainer_state: support.TrainerState,
    fabric: L.Fabric,
    train_factories: dict[K, mechanics.DatasetFactory],
    val_factories: dict[K, mechanics.DatasetFactory],
    vectors: dict[K, None | PrecomputedDsetVectors],
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    search_config: core_config.SearchConfig,
    collate_config: core_config.RetrievalCollateConfig,
    train_dataloader_config: core_config.DataLoaderConfig,
    eval_dataloader_config: core_config.DataLoaderConfig,
    dl_sampler: typing.Optional[DataloaderSampler] = None,
    cache_dir: pathlib.Path,
    serve_on_gpu: bool = False,
    checkpoint_path: typing.Optional[str] = None,
) -> support.TrainerState:
    """Index the sections and train the ranker."""
    barrier_fn = functools.partial(support._barrier_fn, fabric=fabric)

    # Gather datasets and their corresponding vectors
    task = _make_retrieval_task(
        train_factories=train_factories,
        val_factories=val_factories,
        vectors=vectors,
    )
    if fabric.is_global_zero:
        rich.print(task)

    # parameters
    parameters = trainer_state.get_parameters()

    # free GPU resources
    if support.is_engine_enabled(parameters, "faiss"):
        ranker.to("cpu")

    barrier_fn("Init search engines..")
    with search_engine.build_search_engine(
        sections=task.sections.data,
        vectors=task.sections.vectors,
        config=search_config,
        cache_dir=cache_dir,
        faiss_enabled=support.is_engine_enabled(parameters, "faiss"),
        bm25_enabled=support.is_engine_enabled(parameters, "bm25"),
        skip_setup=not fabric.is_global_zero,
        barrier_fn=barrier_fn,
        serve_on_gpu=serve_on_gpu,
    ) as master:
        barrier_fn("index-and-train-search-setup")
        search_client = master.get_client()

        # instantiate the dataloader
        logger.debug("Instantiating dataloader..")
        init_dataloader = functools.partial(
            support.instantiate_retrieval_dataloader,
            sections=task.sections,
            tokenizer=tokenizer,
            search_client=search_client,
            collate_config=collate_config,
            parameters=parameters,
            cache_dir=cache_dir,
            barrier_fn=barrier_fn,
            rank=fabric.global_rank,
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
        logger.debug("Training ranker..")
        ranker = fabric.to_device(ranker)  # type: ignore
        trainer_state = _training_loop(
            ranker=ranker,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
            fabric=fabric,
            train_dl=train_dl,
            val_dl=val_dl,
            checkpoint_path=checkpoint_path,
        )

    return trainer_state


def _training_loop(  # noqa: C901
    *,
    ranker: Ranker,
    optimizer: torch.optim.Optimizer,
    trainer_state: support.TrainerState,
    fabric: L.Fabric,
    train_dl: torch_data.DataLoader,
    val_dl: torch_data.DataLoader,
    scheduler: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_path: typing.Optional[str] = None,
) -> support.TrainerState:
    _check_ranker_and_optimizer(ranker, optimizer)
    optimizer.zero_grad()
    ranker.train()
    train_iter = iter(train_dl)

    # infer the number of training and valid steps
    n_train_steps, n_val_steps = _infer_num_steps(trainer_state, fabric, val_dl)

    with BatchProgressBar(disable=not fabric.is_global_zero) as pbar:
        train_pbar = pbar.add_task(
            f"Training period {1+trainer_state.period}",
            total=n_train_steps,
            info=_pbar_info(trainer_state),
        )
        eval_metrics = None
        for local_step in range(n_train_steps):
            batch = _sample_batch(trainer_state, train_iter=train_iter, train_dl=train_dl)

            # Forward pass
            is_accumulating = local_step % trainer_state.accumulate_grad_batches != 0
            with fabric.no_backward_sync(ranker, enabled=is_accumulating):  # type: ignore
                step_metrics = ranker.training_step(batch)
                loss = step_metrics["loss"]
                fabric.backward(loss / trainer_state.accumulate_grad_batches)

            # Log the training metrics
            if trainer_state.step % trainer_state.log_interval == 0:
                fabric.log_dict(
                    metrics={
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

                # Update the trainer state and the progress bar
                trainer_state.step += 1
                pbar.update(
                    train_pbar,
                    advance=1,
                    info=_pbar_info(
                        trainer_state,
                        train_metrics=step_metrics,
                        eval_metrics=eval_metrics,
                    ),
                )

                # Validation
                if trainer_state.step % trainer_state.val_check_interval == 0:
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

    optimizer.zero_grad()
    return trainer_state


def _check_ranker_and_optimizer(ranker: Ranker, optimizer: torch.optim.Optimizer) -> None:
    if not fabric_wrappers.is_wrapped(ranker):
        raise RuntimeError("Ranker must be wrapped by lightning `Fabric`.")
    if not fabric_wrappers.is_wrapped(optimizer):
        raise RuntimeError("Optimizer must be wrapped by lightning `Fabric`.")


def _infer_num_steps(
    trainer_state: support.TrainerState,
    fabric: L.Fabric,
    val_dl: torch_data.DataLoader,
) -> tuple[int, int]:
    if trainer_state.period_max_steps is None:
        raise ValueError("`trainer_state.period_max_steps` must be set.")
    n_train_steps = trainer_state.period_max_steps - trainer_state.step
    if trainer_state.n_max_eval is None:
        n_val_steps = len(val_dl)
    else:
        eff_eval_bs = fabric.world_size * val_dl.batch_size  # type: ignore
        n_val_steps = min(len(val_dl), max(1, -(-trainer_state.n_max_eval // eff_eval_bs)))
    return n_train_steps, n_val_steps


def _sample_batch(
    trainer_state: support.TrainerState,
    train_iter: typing.Iterator[dict[str, torch.Tensor]],
    train_dl: torch_data.DataLoader,
) -> dict[str, torch.Tensor]:
    try:
        batch = next(train_iter)
    except StopIteration:
        trainer_state.epoch += 1
        train_iter = iter(train_dl)
        batch = next(train_iter)
    return batch


@torch.no_grad()
def _validation_loop(
    ranker: Ranker,
    fabric: L.Fabric,
    trainer_state: support.TrainerState,
    val_dl: torch_data.DataLoader,
    n_steps: int,
    pbar: progress.Progress,
) -> dict[str, float | torch.Tensor]:
    ranker.eval()
    val_pbar = pbar.add_task("Validation", total=n_steps, info=f"{n_steps} steps")
    metrics = collections.defaultdict(list)
    for batch in val_dl:
        output = ranker.validation_step(batch)
        pbar.update(val_pbar, advance=1, taskinfo=_pbar_info(trainer_state, output))
        for k, v in output.items():
            metrics[k].append(_format_metric(v))
        if trainer_state.step >= n_steps:
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

    def __init__(self) -> None:
        self._laps = []
        _start_time = None

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
        self._start_time = None
        return self

    def get_total_time(self) -> float:
        """Return the total time elapsed since the chrono was started."""
        return sum(end - start for start, end in self._laps)

    def get_avg_time(self) -> float:
        """Return the average time elapsed since the chrono was started."""
        return self.get_total_time() / len(self._laps)


def _extract_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {f"lr_{i}": param_group["lr"] for i, param_group in enumerate(optimizer.param_groups)}


def _pbar_info(
    state: support.TrainerState,
    train_metrics: typing.Optional[dict[str, typing.Any]] = None,
    eval_metrics: typing.Optional[dict[str, typing.Any]] = None,
) -> str:
    desc = f"step={1+state.step}/{state.period_max_steps}/{state.max_steps} • epoch={1+state.epoch}"
    if train_metrics or eval_metrics:
        suppl = []
        if train_metrics is not None:
            suppl += [f"train/loss={train_metrics['loss']:.3f}"]
        if eval_metrics is not None:
            suppl += [f"val/loss={eval_metrics['loss']:.3f}"]

        desc = f"[bold yellow]{', '.join(suppl)}[/bold yellow] • {desc}"

    return desc


def _make_retrieval_task(
    train_factories: dict[K, mechanics.DatasetFactory],
    val_factories: dict[K, mechanics.DatasetFactory],
    vectors: dict[K, PrecomputedDsetVectors | None],
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
        train_questions=support.concatenate_datasets(
            [
                support.DsetWithVectors.cast(data=factory.get_qa_split(), vectors=_vec(key, "question"))
                for key, factory in train_factories.items()
            ]
        ),
        val_questions=support.concatenate_datasets(
            [
                support.DsetWithVectors.cast(data=factory.get_qa_split(), vectors=_vec(key, "question"))
                for key, factory in val_factories.items()
            ]
        ),
        sections=support.concatenate_datasets(
            [
                support.DsetWithVectors.cast(data=factory.get_sections(), vectors=_vec(key, "section"))
                for key, factory in {**train_factories, **val_factories}.items()
            ]
        ),
    )
