import functools
import pathlib

import lightning as L
import loguru
import torch
import vod_configs
import vod_types as vt
from vod_dataloaders.tokenizer_collate import TokenizerCollate
from vod_ops.utils import helpers
from vod_tools.ts_factory.ts_factory import TensorStoreFactory

from .predict import Predict


def compute_vectors(  # noqa: PLR0913
    dataset: vt.DictsSequence,
    *,
    module: torch.nn.Module,
    fabric: L.Fabric,
    collate_config: vod_configs.TokenizerCollateConfig,
    dataloader_config: vod_configs.DataLoaderConfig,
    save_dir: pathlib.Path,
    field: str,
    validate_store: int = 1_000,
    locator: str = "embedding",
) -> TensorStoreFactory:
    """Compute the vectors for a given dataset and field. Hanldes distributed execution on a single node."""
    collate_fn = TokenizerCollate.instantiate(collate_config, field=field)
    barrier_fn = functools.partial(helpers.barrier_fn, fabric=fabric)

    # construct the `predict` function
    predict_fn = Predict(
        dataset=dataset,  # type: ignore
        save_dir=save_dir,
        model=module,
        collate_fn=collate_fn,
        model_output_key={"query": "query_encoding", "section": "section_encoding"}[field],
    )

    # check if the store already exists, validate and read it if it does.
    # if the validation fails, delete the store and recompute the vectors.
    if predict_fn.exists():
        loguru.logger.info(f"{locator} - Found pre-computed vectors at `{predict_fn.store_path}`")
        is_valid = predict_fn.validate_store(validate_store, raise_exc=False)
        barrier_fn(f"{locator} - Validated existing vector store.")
        if is_valid:
            loguru.logger.debug(f"{locator} - Loading `{predict_fn.store_path}`")
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
