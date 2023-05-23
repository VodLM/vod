from __future__ import annotations

import dataclasses
import functools
import pathlib
import typing

import lightning as L
import rich
import transformers
from loguru import logger

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core import mechanics
from raffle_ds_research.core.mechanics import search_engine
from raffle_ds_research.core.ml.ranker import Ranker
from raffle_ds_research.core.workflows.precompute import PrecomputedDsetVectors
from raffle_ds_research.core.workflows.utils import support
from raffle_ds_research.tools import dstruct

K = typing.TypeVar("K")


@dataclasses.dataclass(frozen=True)
class RetrievalTask:
    """Holds the train and validation datasets."""

    train_questions: support.DsetWithVectors
    val_questions: support.DsetWithVectors
    sections: support.DsetWithVectors


def index_and_train(
    ranker: Ranker,
    *,
    trainer: L.Trainer,
    train_factories: dict[K, mechanics.DatasetFactory],
    val_factories: dict[K, mechanics.DatasetFactory],
    vectors: dict[K, None | PrecomputedDsetVectors],
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    search_config: core_config.SearchConfig,
    collate_config: core_config.RetrievalCollateConfig,
    train_dataloader_config: core_config.DataLoaderConfig,
    eval_dataloader_config: core_config.DataLoaderConfig,
    cache_dir: pathlib.Path,
    parameters: typing.Optional[dict[str, float]] = None,
    serve_on_gpu: bool = False,
) -> Ranker:
    """Index the sections and train the ranker."""
    barrier_fn = functools.partial(support._barrier_fn, trainer=trainer)
    task = _make_retrieval_task(
        train_factories=train_factories,
        val_factories=val_factories,
        vectors=vectors,
    )
    if trainer.is_global_zero:
        rich.print(task)

    trainer.strategy.barrier("Init search engines..")
    with search_engine.build_search_engine(
        sections=task.sections.data,
        vectors=task.sections.vectors,
        config=search_config,
        cache_dir=cache_dir,
        faiss_enabled=support.is_engine_enabled(parameters, "faiss"),
        bm25_enabled=support.is_engine_enabled(parameters, "bm25"),
        skip_setup=not trainer.is_global_zero,
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
            rank=trainer.global_rank,
        )

        # train the ranker
        logger.debug("Training ranker..")
        trainer.fit(
            ranker,
            train_dataloaders=init_dataloader(
                questions=task.train_questions,
                dataloader_config=train_dataloader_config,
            ),
            val_dataloaders=init_dataloader(
                questions=task.val_questions,
                dataloader_config=eval_dataloader_config,
            ),
        )

    return ranker


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
