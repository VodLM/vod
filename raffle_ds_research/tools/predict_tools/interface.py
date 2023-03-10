from __future__ import annotations

import math
import shutil
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import datasets
import numpy as np
import pytorch_lightning as pl
import tensorstore
import torch
from loguru import logger
from omegaconf import DictConfig, omegaconf
from pydantic import BaseModel, Extra, validator
from pytorch_lightning import Trainer

from raffle_ds_research.tools.dataset_builder.builder import CollateFnProtocol, DatasetProtocol
from raffle_ds_research.tools.utils.tensor_tools import serialize_tensor

from .callback import StorePredictions
from .ts_utils import TensorStoreFactory
from .wrappers import _warp_as_lightning_model, _wrap_collate_fn_with_indices, _wrap_dataset_with_indices


class DataLoaderForPredictKwargs(BaseModel):
    class Config:
        extra = Extra.forbid

    batch_size: int
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    timeout: int = 0
    worker_init_fn: Callable = None
    prefetch_factor: int = 2
    persistent_workers: bool = False

    @validator("drop_last", pre=True)
    def _force_drop_last(cls, v):
        if v:
            logger.warning(
                "drop_last is set to True. "
                "This might result in processing only part of the dataset. "
                "Forcing `drop_last` to False."
            )
        return False

    @validator("shuffle", pre=True)
    def _force_shuffle(cls, v):
        if v:
            logger.debug(
                "Shuffle is set to True. " "This is unnecessary for predictions. " "Forcing `shuffle` to False."
            )
        return False


def predict(
    dataset: DatasetProtocol | datasets.Dataset | datasets.DatasetDict | dict[str, DatasetProtocol],
    *,
    trainer: Trainer,
    cache_dir: str | Path,
    model: torch.nn.Module | pl.LightningModule,
    collate_fn: CollateFnProtocol = torch.utils.data.dataloader.default_collate,
    model_output_key: Optional[str] = None,
    loader_kwargs: Optional[dict[str, Any] | DictConfig | DataLoaderForPredictKwargs] = None,
    ts_kwargs: Optional[dict[str, Any]] = None,
    validate_store: bool | int = True,
) -> TensorStoreFactory | dict[str, TensorStoreFactory]:
    """Compute predictions for a dataset and store them in a tensorstore"""
    if isinstance(dataset, (dict, datasets.DatasetDict)):
        return _predict_dict(
            dataset,
            trainer=trainer,
            cache_dir=cache_dir,
            model=model,
            collate_fn=collate_fn,
            model_output_key=model_output_key,
            loader_kwargs=loader_kwargs,
            ts_kwargs=ts_kwargs,
            validate_store=validate_store,
        )
    else:
        return _predict_single(
            dataset,
            trainer=trainer,
            cache_dir=cache_dir,
            model=model,
            collate_fn=collate_fn,
            model_output_key=model_output_key,
            loader_kwargs=loader_kwargs,
            ts_kwargs=ts_kwargs,
            validate_store=validate_store,
        )


def _predict_dict(
    dataset: dict[str, DatasetProtocol],
    **kwargs,
) -> dict[str, TensorStoreFactory]:
    """Compute predictions for a dataset and store them in a tensorstore"""
    return {split: _predict_single(dset, **kwargs) for split, dset in dataset.items()}


def _predict_single(
    dataset: DatasetProtocol | datasets.Dataset,
    *,
    trainer: Trainer,
    cache_dir: str | Path,
    model: torch.nn.Module | pl.LightningModule,
    collate_fn: CollateFnProtocol = torch.utils.data.dataloader.default_collate,
    model_output_key: Optional[str] = None,
    loader_kwargs: Optional[dict[str, Any] | DictConfig | DataLoaderForPredictKwargs] = None,
    ts_kwargs: Optional[dict[str, Any]] = None,
    validate_store: bool | int = True,
) -> TensorStoreFactory:
    """Compute predictions for a dataset and store them in a tensorstore"""

    # set the fingerprint and define the store path
    dset_fingerprint = _get_dset_fingerprint(dataset)
    model_fingerprint = _get_model_fingerprint(model)
    collate_fn_fingerprint = _get_collate_fn_fingerprint(collate_fn)
    logger.debug(f"Computing vectors for dataset: {dset_fingerprint} and model: {model_fingerprint}")
    fingerprint = f"{dset_fingerprint}_{model_fingerprint}_{collate_fn_fingerprint}"
    if model_output_key:
        fingerprint += f"_{model_output_key}"

    index_path = Path(cache_dir, "predictions", f"{fingerprint}.ts")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # infer the dataset shape by running one example through the model
    vector_shape = _infer_vector_shape(model, model_output_key, dataset=dataset, collate_fn=collate_fn)

    # build the store config and open the store
    ts_kwargs = ts_kwargs or {}
    dset_shape = (len(dataset), *vector_shape)
    ts_config: TensorStoreFactory = TensorStoreFactory.from_factory(
        path=index_path,
        shape=dset_shape,
        **ts_kwargs,
    )
    if index_path.exists():
        logger.info(f"Loading TensorStore from path `{index_path}`. Shape={dset_shape}")
        store = ts_config.open(create=False)
    else:
        logger.info(f"creating TensorStore at path `{index_path}`. Shape={dset_shape}")
        store = ts_config.open(create=True, delete_existing=False)

        # compute the predictions
        try:
            _compute_and_store_predictions(
                trainer=trainer,
                dataset=dataset,
                model=model,
                collate_fn=collate_fn,
                store=store,
                loader_kwargs=loader_kwargs,
                model_output_key=model_output_key,
            )
        except KeyboardInterrupt:
            logger.warning(f"Prediction step was keyboard-interrupted. Deleting store.")
            shutil.rmtree(index_path)
        except Exception as e:
            del store
            raise e

    # validate that all store values have been initialized
    if validate_store:
        n_samples = math.inf if isinstance(validate_store, bool) else validate_store
        zero_ids = list(_get_zero_indices(dataset, store, n_samples=n_samples))
        if len(zero_ids):
            raise ValueError(
                f"Vector at indices {zero_ids} are all zeros. "
                f"This happens if the store has been initialized but not updated with predictions. "
                f"Please delete the store at {index_path} and try again. "
                f"NB: this could happen if the model outputs a zero vector."
            )

    # close the store and return the config
    del store
    return ts_config


def _infer_vector_shape(
    model: torch.nn.Module | pl.LightningModule,
    model_output_key: Optional[str],
    *,
    dataset: DatasetProtocol,
    collate_fn: CollateFnProtocol,
) -> tuple[int, ...]:
    # todo: handle variable length inputs
    try:
        vector_shape = model.get_output_shape(model_output_key)
    except AttributeError as exc:
        logger.debug(
            f"{exc}. "
            f"Inferring the vector size by running one example through the model. "
            f"Implement `model.get_output_shape(output_key: str) -> tuple[int,...]` to skip this step."
        )
        batch = collate_fn([dataset[0]])
        if hasattr(model, "predict"):
            one_vec = model.predict(batch)
        else:
            one_vec = model(batch)
        if model_output_key is not None:
            one_vec = one_vec[model_output_key]
        vector_shape = one_vec.shape[1:]

    return vector_shape


def _get_zero_indices(dataset, store, n_samples: Number) -> Iterable[int]:
    if n_samples < len(dataset):
        ids = np.random.choice(len(dataset), int(n_samples), replace=False)
    else:
        ids = range(len(dataset))
    for i in ids:
        vec = store[i].read()
        if np.all(vec == 0):
            yield i


@torch.inference_mode()
def _compute_and_store_predictions(
    trainer: pl.Trainer,
    dataset: DatasetProtocol,
    model: torch.nn.Module | pl.LightningModule,
    collate_fn: Callable[[list[dict]], dict],
    store: tensorstore.TensorStore,
    loader_kwargs: Optional[dict[str, Any] | DictConfig | DataLoaderForPredictKwargs] = None,
    model_output_key: Optional[str] = None,
) -> tensorstore.TensorStore:
    """Compute predictions for a dataset and store them in a tensorstore"""
    # wrap the input variables
    dset_with_ids = _wrap_dataset_with_indices(dataset)
    collate_fn_with_ids = _wrap_collate_fn_with_indices(collate_fn)
    pl_model = _warp_as_lightning_model(model)

    # build the dataloader
    if isinstance(loader_kwargs, DictConfig):
        loader_kwargs = omegaconf.OmegaConf.to_container(loader_kwargs, resolve=True)
    if not isinstance(loader_kwargs, DataLoaderForPredictKwargs):
        loader_kwargs = loader_kwargs or {}
        if len(loader_kwargs) == 0:
            loader_kwargs = {"batch_size": 10}
            logger.warning("No `loader_kwargs` were provided. Using default batch_size=10. ")
        loader_kwargs = DataLoaderForPredictKwargs(**loader_kwargs)
    loader = torch.utils.data.DataLoader(dset_with_ids, collate_fn=collate_fn_with_ids, **loader_kwargs.dict())

    # process the dataset and store the predictions in the tensorstore
    with StorePredictions(trainer, store, model_output_key=model_output_key):
        trainer.predict(pl_model, dataloaders=loader)

    return store


def _get_dset_fingerprint(dataset: DatasetProtocol) -> str:
    N_MAX_SAMPLES = 1000
    if isinstance(dataset, datasets.Dataset):
        return dataset._fingerprint
    else:
        hasher = datasets.fingerprint.Hasher()
        hasher.update({"length": len(dataset)})

        # select random row ids
        if len(dataset) > N_MAX_SAMPLES:
            rgn = np.random.RandomState(0)
            ids = rgn.choice(len(dataset), size=N_MAX_SAMPLES, replace=False)
        else:
            ids = range(len(dataset))

        # hash the rows
        for i in ids:
            hasher.update(dataset[i])

        return hasher.hexdigest()


def _get_model_fingerprint(model: torch.nn.Module):
    state = model.state_dict()
    hasher = datasets.fingerprint.Hasher()
    hasher.update(type(model).__name__)
    for k, v in sorted(state.items(), key=lambda x: x[0]):
        hasher.update(k)
        u = serialize_tensor(v)
        hasher.update(u)
    return hasher.hexdigest()


def _get_collate_fn_fingerprint(collate_fn: CollateFnProtocol) -> str:
    hasher = datasets.fingerprint.Hasher()
    hasher.update(collate_fn)
    return hasher.hexdigest()

    pass
