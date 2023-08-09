from __future__ import annotations

import functools
import pathlib
from typing import Any

import datasets
import lightning as L
import loguru
import transformers
from vod_tools import dstruct, pipes, predict
from vod_workflows.utils import helpers

from src import vod_configs, vod_models


def compute_vectors(
    dataset: dstruct.SizedDataset[dict[str, Any]] | datasets.Dataset,
    *,
    ranker: vod_models.Ranker,
    fabric: L.Fabric,
    tokenizer: transformers.PreTrainedTokenizerBase,
    collate_config: vod_configs.BaseCollateConfig,
    dataloader_config: vod_configs.DataLoaderConfig,
    cache_dir: pathlib.Path,
    field: str,
    validate_store: int = 1_000,
    locator: None | str = None,
) -> dstruct.TensorStoreFactory:
    """Compute the vectors for a given dataset and field. Hanldes distributed execution on a single node."""
    collate_fn = init_predict_collate_fn(collate_config, field=field, tokenizer=tokenizer)
    barrier_fn = functools.partial(helpers.barrier_fn, fabric=fabric)

    # construct the `predict` function
    predict_fn = predict.Predict(
        dataset=dataset,  # type: ignore
        cache_dir=cache_dir,
        model=ranker,
        collate_fn=collate_fn,
        model_output_key={"query": "hq", "section": "hd"}[field],
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
        "query": config.query_max_length,
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
