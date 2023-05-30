from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Any, Callable, Literal, TypeVar

import lightning as L  # noqa: N812
import loguru
import transformers

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core.mechanics import dataset_factory
from raffle_ds_research.core.ml import Ranker
from raffle_ds_research.core.workflows.utils import support
from raffle_ds_research.tools import dstruct, pipes, predict_tools

K = TypeVar("K")


def _get_key(x: tuple) -> str:
    return str(x[0])


@dataclasses.dataclass(frozen=True)
class PrecomputedDsetVectors:
    """Holds the vectors for a given dataset and field."""

    questions: dstruct.TensorStoreFactory
    sections: dstruct.TensorStoreFactory


def compute_vectors(
    factories: dict[K, dataset_factory.DatasetFactory],
    *,
    ranker: Ranker,
    trainer: L.Trainer,
    cache_dir: pathlib.Path,
    dataset_config: core_config.MultiDatasetFactoryConfig,
    collate_config: core_config.BaseCollateConfig,
    dataloader_config: core_config.DataLoaderConfig,
) -> dict[K, PrecomputedDsetVectors]:
    """Compute the vectors for a collection of datasets."""
    predict_fn: Callable[..., dstruct.TensorStoreFactory] = functools.partial(
        compute_dataset_vectors,
        ranker=ranker,
        trainer=trainer,
        collate_config=collate_config,
        dataloader_config=dataloader_config,
        tokenizer=dataset_config.tokenizer,
        cache_dir=cache_dir,
    )

    # compute the vectors for each dataset
    question_vecs = {
        key: predict_fn(factory, field="question") for key, factory in sorted(factories.items(), key=_get_key)
    }
    section_vecs = {
        key: predict_fn(factory, field="section") for key, factory in sorted(factories.items(), key=_get_key)
    }

    # format the output and return
    return {key: PrecomputedDsetVectors(questions=question_vecs[key], sections=section_vecs[key]) for key in factories}


def compute_dataset_vectors(
    factory: dataset_factory.DatasetFactory | dstruct.SizedDataset[dict[str, Any]],
    *,
    ranker: Ranker,
    trainer: L.Trainer,
    tokenizer: transformers.PreTrainedTokenizerBase,
    collate_config: core_config.BaseCollateConfig,
    dataloader_config: core_config.DataLoaderConfig,
    cache_dir: pathlib.Path,
    field: Literal["question", "section"] = "question",
    validate_store: int = 1_000,
) -> dstruct.TensorStoreFactory:
    """Compute the vectors for a given dataset and field."""
    model_output_key = {"question": "hq", "section": "hd"}[field]
    collate_fn = init_predict_collate_fn(collate_config, field=field, tokenizer=tokenizer)
    barrier_fn = functools.partial(support._barrier_fn, trainer=trainer)

    # construct the dataset
    if isinstance(factory, dataset_factory.DatasetFactory):
        dataset = factory(what=field)
        locator = f"{factory.name}:{factory.split}({field})"
    else:
        dataset = factory
        locator = type(dataset).__name__

    # construct the `predict` function
    predict_fn = predict_tools.Predict(
        dataset=dataset,  # type: ignore
        cache_dir=cache_dir,
        model=ranker,
        model_output_key=model_output_key,
        collate_fn=collate_fn,
    )

    # check if the store already exists, validate and read it if it does.
    # if the validation fails, delete the store and recompute the vectors.
    if predict_fn.exists():
        loguru.logger.info(f"{locator} - Found pre-computed vectors.")
        if predict_fn.validate_store(validate_store, raise_exc=False):
            loguru.logger.debug(f"{locator} - loading `{predict_fn.store_path}`")
            return predict_fn.read()
        if trainer.is_global_zero:
            loguru.logger.warning(f"{locator} - Invalid store. Deleting it..")
            predict_fn.rm()

    # create the store
    barrier_fn(f"{locator} - Checked existing vector store.")
    if trainer.is_global_zero:
        loguru.logger.info(f"{locator} - Instantiating vector store at `{predict_fn.store_path}`")
        predict_fn.instantiate()

    # compute the vectors
    barrier_fn(f"{locator} - About to compute vectors.")
    return predict_fn(
        trainer=trainer,
        loader_kwargs=dataloader_config,
        validate_store=validate_store,
        open_mode="a",
    )  # type: ignore


def init_predict_collate_fn(
    config: core_config.BaseCollateConfig,
    *,
    field: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> pipes.Collate:
    """Initialize the collate function for the `predict` function."""
    max_length = {
        "question": config.question_max_length,
        "section": config.section_max_length,
    }[field]

    # init the collate_fn
    return functools.partial(
        pipes.torch_tokenize_collate,
        tokenizer=tokenizer,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        prefix_key=f"{field}.",
    )
