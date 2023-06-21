from __future__ import annotations

import dataclasses
import functools
import pathlib
import typing

import lightning as L
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
from raffle_ds_research.core.workflows.utils import support
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
) -> support.TrainerState:
    """Index the sections and train the ranker."""
    barrier_fn = functools.partial(support._barrier_fn, fabric=fabric)
    task = _make_retrieval_task(
        train_factories=train_factories,
        val_factories=val_factories,
        vectors=vectors,
    )
    if fabric.is_global_zero:
        rich.print(task)

    barrier_fn("Init search engines..")
    parameters = trainer_state.get_parameters()
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
        trainer_state = _training_loop(
            ranker=ranker,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
            fabric=fabric,
            train_dl=train_dl,
            val_dl=val_dl,
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
) -> support.TrainerState:
    if not fabric_wrappers.is_wrapped(ranker):
        raise ValueError("Ranker must be wrapped by lightning fabric.")
    if not fabric_wrappers.is_wrapped(optimizer):
        raise ValueError("Optimizer must be wrapped by lightning fabric.")
    optimizer.zero_grad()
    ranker.train()
    train_iter = iter(train_dl)

    # infer the number of training and valid steps
    n_train_steps = trainer_state.accumulate_grad_batches * (trainer_state.period_max_steps - trainer_state.step)
    if trainer_state.limit_val_batches is None:
        n_val_steps = len(val_dl)
    else:
        n_val_steps = min(len(val_dl), max(1, trainer_state.limit_val_batches // fabric.world_size))

    with BatchProgressBar(disable=not fabric.is_global_zero) as pbar:
        train_pbar = pbar.add_task(
            f"Training period {1+trainer_state.period}",
            total=n_train_steps,
            info=_pbar_info(trainer_state),
        )
        for local_step in range(n_train_steps):
            # sample a batch
            try:
                batch = next(train_iter)
            except StopIteration:
                trainer_state = dataclasses.replace(trainer_state, epoch=trainer_state.epoch + 1)
                train_iter = iter(train_dl)
                batch = next(train_iter)

            # forward pass
            is_accumulating = local_step % trainer_state.accumulate_grad_batches != 0
            with fabric.no_backward_sync(ranker, enabled=is_accumulating):  # type: ignore
                output = ranker.training_step(batch)
                loss = output["loss"]
                fabric.backward(loss / trainer_state.accumulate_grad_batches)

            # logging
            if trainer_state.step % trainer_state.log_interval == 0:
                fabric.log_dict(
                    metrics=output,
                    step=trainer_state.step,
                )

            # optimization step
            if not is_accumulating:
                # clip the gradients
                if trainer_state.gradient_clip_val:
                    fabric.clip_gradients(ranker, optimizer, max_norm=trainer_state.gradient_clip_val)

                # run an optimization step, reset the gradients and update the learning rate
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                trainer_state = dataclasses.replace(trainer_state, step=trainer_state.step + 1)
                pbar.update(train_pbar, advance=1, taskinfo=_pbar_info(trainer_state, output))

                # validation
                if trainer_state.step % trainer_state.val_check_interval == 0:
                    _validation_loop(
                        ranker=ranker,
                        fabric=fabric,
                        trainer_state=trainer_state,
                        val_dl=val_dl,
                        n_steps=n_val_steps,
                        pbar=pbar,
                    )
                    ranker.train()

    return trainer_state


@torch.no_grad()
def _validation_loop(
    ranker: Ranker,
    fabric: L.Fabric,
    trainer_state: support.TrainerState,
    val_dl: torch_data.DataLoader,
    n_steps: int,
    pbar: progress.Progress,
) -> None:
    ranker.eval()
    val_pbar = pbar.add_task("Validation", total=n_steps, info=f"n_steps={n_steps}")
    for batch in val_dl:
        output = ranker.validation_step(batch)
        fabric.log_dict(
            metrics=output,
            step=trainer_state.step,
        )
        pbar.update(val_pbar, advance=1, taskinfo=_pbar_info(trainer_state, output))
        if trainer_state.step >= n_steps:
            break

    pbar.remove_task(val_pbar)


def _pbar_info(
    state: support.TrainerState,
    outputs: typing.Optional[dict[str, typing.Any]] = None,
) -> str:
    desc = f"step={1+state.step}/{state.period_max_steps}/{state.max_steps} • epoch={1+state.epoch}"
    if outputs is not None:
        desc = f"[yellow bold]loss={outputs['loss']:.3f}[/yellow bold] • " + desc

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
