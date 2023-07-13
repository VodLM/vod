from __future__ import annotations

import functools
import pathlib
from typing import Any, Callable, Literal, TypeVar

import lightning as L
import loguru
import transformers
from vod_tools import dstruct, pipes, predict
from vod_workflows.utils import helpers

from src import vod_configs, vod_datasets, vod_models

K = TypeVar("K")


def _get_key(x: tuple) -> str:
    return str(x[0])


def compute_vectors(
    factories: dict[K, vod_datasets.DatasetFactory],
    *,
    ranker: vod_models.Ranker,
    tokenizer: transformers.PreTrainedTokenizerBase,
    fabric: L.Fabric,
    cache_dir: pathlib.Path,
    collate_config: vod_configs.BaseCollateConfig,
    dataloader_config: vod_configs.DataLoaderConfig,
) -> dict[K, helpers.PrecomputedDsetVectors]:
    """Compute the vectors for a collection of datasets."""
    predict_fn: Callable[..., dstruct.TensorStoreFactory] = functools.partial(
        compute_dataset_vectors,
        ranker=ranker,
        fabric=fabric,
        collate_config=collate_config,
        dataloader_config=dataloader_config,
        tokenizer=tokenizer,
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
    return {
        key: helpers.PrecomputedDsetVectors(questions=question_vecs[key], sections=section_vecs[key])
        for key in factories
    }


def compute_dataset_vectors(
    factory: vod_datasets.DatasetFactory | dstruct.SizedDataset[dict[str, Any]],
    *,
    ranker: vod_models.Ranker,
    fabric: L.Fabric,
    tokenizer: transformers.PreTrainedTokenizerBase,
    collate_config: vod_configs.BaseCollateConfig,
    dataloader_config: vod_configs.DataLoaderConfig,
    cache_dir: pathlib.Path,
    field: Literal["question", "section"] = "question",
    validate_store: int = 1_000,
) -> dstruct.TensorStoreFactory:
    """Compute the vectors for a given dataset and field."""
    collate_fn = init_predict_collate_fn(collate_config, field=field, tokenizer=tokenizer)
    barrier_fn = functools.partial(helpers.barrier_fn, fabric=fabric)

    # construct the dataset
    if isinstance(factory, vod_datasets.DatasetFactory):
        dataset = factory(what=field)
        locator = f"{factory.name}:{factory.split}({field})"
    else:
        dataset = factory
        locator = type(dataset).__name__

    # construct the `predict` function
    predict_fn = predict.Predict(
        dataset=dataset,  # type: ignore
        cache_dir=cache_dir,
        model=ranker,
        collate_fn=collate_fn,
        model_output_key={"question": "hq", "section": "hd"}[field],
    )

    # check if the store already exists, validate and read it if it does.
    # if the validation fails, delete the store and recompute the vectors.
    if predict_fn.exists():
        loguru.logger.info(f"{locator} - Found pre-computed vectors.")
        is_valid = predict_fn.validate_store(validate_store, raise_exc=False)
        barrier_fn(f"{locator} - Validated existing vector store.")
        if is_valid:
            loguru.logger.debug(f"{locator} - loading `{predict_fn.store_path}`")
            return predict_fn.read()
        if fabric.is_global_zero:
            loguru.logger.warning(f"{locator} - Invalid store. Deleting it..")
            predict_fn.rm()

    # create the store
    barrier_fn(f"{locator} - Checked existing vector store.")
    if fabric.is_global_zero:
        loguru.logger.info(f"{locator} - Instantiating vector store at `{predict_fn.store_path}`")
        predict_fn.instantiate()

    # compute the vectors
    barrier_fn(f"{locator} - About to compute vectors.")
    return predict_fn(
        fabric=fabric,
        loader_kwargs=dataloader_config,
        validate_store=validate_store,
        open_mode="a",
    )  # type: ignore


def init_predict_collate_fn(
    config: vod_configs.BaseCollateConfig,
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
