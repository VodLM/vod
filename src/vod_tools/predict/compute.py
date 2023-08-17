from __future__ import annotations

from asyncio import Future
from typing import Any, List, Optional, Union

import lightning as L
import omegaconf
import pydantic
import tensorstore as ts
import torch
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger
from omegaconf import DictConfig
from torch.utils import data as torch_data
from typing_extensions import TypeAlias
from vod_tools import dstruct, pipes
from vod_tools.misc.progress import IterProgressBar

from src import vod_configs

from .wrappers import PREDICT_IDX_COL_NAME, CollateWithIndices, DatasetWithIndices

LoaderKwargs: TypeAlias = Union[dict[str, Any], DictConfig, vod_configs.DataLoaderConfig]


class DataLoaderForPredictKwargs(vod_configs.DataLoaderConfig):
    """Confiuguration for `torch.utils.data.Dataloader` for predictions."""

    @pydantic.validator("shuffle", pre=True)
    def _force_shuffle(cls, value: bool) -> bool:
        if value:
            logger.debug("Shuffle is set to True. This is unnecessary for predictions. Forcing `shuffle` to False.")
        return False

    @classmethod
    def instantiate(cls, config: None | LoaderKwargs) -> DataLoaderForPredictKwargs:  # noqa: ANN102
        """Instantiate a `DataLoaderForPredictKwargs` from a config."""
        config = config or {}
        if isinstance(config, DictConfig):
            config = omegaconf.OmegaConf.to_container(config, resolve=True)  # type: ignore
        if isinstance(config, DataLoaderForPredictKwargs):
            return config

        if isinstance(config, pydantic.BaseModel):
            config = config.dict()
        if config is None or len(config) == 0:
            config = {"batch_size": 10}
            logger.warning("No loader config was provided. Using default batch_size=10. ")
        return DataLoaderForPredictKwargs(**config)  # type: ignore


def compute_and_store_predictions(
    fabric: L.Fabric,
    dataset: dstruct.SizedDataset,
    model: torch.nn.Module,
    collate_fn: pipes.Collate,
    store: ts.TensorStore,
    loader_kwargs: Optional[LoaderKwargs] = None,
    model_output_key: Optional[str] = None,
) -> ts.TensorStore:
    """Compute predictions for a dataset and store them in a tensorstore."""
    if not fabric_wrappers.is_wrapped(model):
        raise ValueError("The model must be wrapped with `lightning.fabric.wrappers`.")

    # wrap the dataset and collate_fn to include the indices
    dset_with_ids: dstruct.SizedDataset[dict] = DatasetWithIndices[dict](dataset)
    collate_fn_with_ids: pipes.Collate = CollateWithIndices(collate_fn)

    # build the dataloader
    loader_kwargs = DataLoaderForPredictKwargs.instantiate(loader_kwargs)
    loader = torch_data.DataLoader(
        dset_with_ids,  # type: ignore
        collate_fn=collate_fn_with_ids,
        **loader_kwargs.dict(),
    )

    # process the dataset and store the predictions in the tensorstore
    model.eval()
    return _predict_loop(fabric=fabric, dataloader=loader, store=store, model=model, model_output_key=model_output_key)


@torch.no_grad()
def _predict_loop(
    *,
    fabric: L.Fabric,
    dataloader: torch_data.DataLoader,
    store: ts.TensorStore,
    model: torch.nn.Module,
    model_output_key: Optional[str] = None,
) -> ts.TensorStore:
    """Run predictions and store the output in a `TensorStore`."""
    dataloader_ = fabric.setup_dataloaders(dataloader)
    with IterProgressBar(disable=not fabric.is_global_zero) as pbar:
        task = pbar.add_task(
            "Computing vectors",
            total=len(dataloader_),
            info="Starting...",
        )
        counter = 0
        for batch in dataloader_:
            indices = batch[PREDICT_IDX_COL_NAME]  # type: ignore
            outputs = model(batch)
            vectors = _select_vector_from_output(outputs, model_output_key)
            _write_vectors_to_store(
                store,
                vectors=vectors,
                idx=indices,
                asynchronous=False,
            )
            counter += len(indices)
            pbar.update(
                task,
                advance=1,
                info=f"{fabric.world_size * counter} / {store.shape[0]}",
            )

    return store


def _write_vectors_to_store(
    store: ts.TensorStore,
    vectors: torch.Tensor,
    idx: List[int],
    asynchronous: bool = False,
) -> Optional[Future]:
    """Write vectors to a `TensorStore`."""
    if idx is None:
        raise ValueError("idx must be provided")
    if vectors.dtype == torch.bfloat16:
        vectors = vectors.to(torch.float32)
    vectors = vectors.detach().cpu().numpy()
    dtype = store.spec().dtype
    vectors = vectors.astype(dtype.numpy_dtype)

    if asynchronous:
        return store[idx].write(vectors)

    store[idx] = vectors
    return None


def _select_vector_from_output(batch: torch.Tensor | dict, key: Optional[str] = None) -> torch.Tensor:
    if key is None:
        if isinstance(batch, dict):
            raise TypeError("Input batch is a dictionary, the argument `field` must be set.")
        return batch

    if key not in batch:
        raise ValueError(f"Key {key} not found in batch. Found {batch.keys()}")

    return batch[key]
