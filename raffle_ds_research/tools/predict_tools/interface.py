# pylint: disable=protected-access
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional

import datasets
import lightning.pytorch as pl
import numpy as np
import pydantic
import tensorstore
import torch
from datasets import fingerprint
from loguru import logger
from omegaconf import DictConfig, omegaconf
from rich.progress import track

from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.dataset_builder.builder import DatasetProtocol
from raffle_ds_research.tools.utils import loader_config
from raffle_ds_research.tools.utils.tensor_tools import serialize_tensor

from .callback import StorePredictions
from .ts_utils import TensorStoreFactory
from .wrappers import CollateWithIndices, DatasetWithIndices, _warp_as_lightning_model


class DataLoaderForPredictKwargs(loader_config.DataLoaderConfig):
    """Confiuguration for `torch.utils.data.Dataloader` for predictions."""

    @pydantic.validator("shuffle", pre=True)
    def _force_shuffle(cls, value: bool) -> bool:
        if value:
            logger.debug("Shuffle is set to True. This is unnecessary for predictions. Forcing `shuffle` to False.")
        return False


def predict(
    dataset: DatasetProtocol | datasets.Dataset | datasets.DatasetDict | dict[str, DatasetProtocol],
    *,
    cache_dir: str | Path,
    model: torch.nn.Module | pl.LightningModule,
    collate_fn: pipes.Collate = torch.utils.data.dataloader.default_collate,
    trainer: Optional[pl.Trainer] = None,
    model_output_key: Optional[str] = None,
    loader_kwargs: Optional[dict[str, Any] | DictConfig | loader_config.DataLoaderConfig] = None,
    ts_kwargs: Optional[dict[str, Any]] = None,
    validate_store: bool | int = True,
    read_only: bool = False,
) -> TensorStoreFactory | dict[str, TensorStoreFactory]:
    """Compute predictions for a dataset and store them in a tensorstore.

    Note: the fingerprint of the collate_fn changes after a call of `_predict_single`. Can't figure out why.
    """
    if trainer is None:
        trainer = pl.Trainer()
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
            read_only=read_only,
        )

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
        read_only=read_only,
    )


def _predict_dict(
    dataset: dict[str, DatasetProtocol],
    *,
    trainer: pl.Trainer,
    cache_dir: str | Path,
    model: torch.nn.Module | pl.LightningModule,
    collate_fn: pipes.Collate,
    model_output_key: Optional[str] = None,
    loader_kwargs: Optional[dict[str, Any] | DictConfig | loader_config.DataLoaderConfig] = None,
    ts_kwargs: Optional[dict[str, Any]] = None,
    validate_store: bool | int = True,
    read_only: bool = False,
) -> dict[str, TensorStoreFactory]:
    """Compute predictions for a dataset and store them in a tensorstore."""
    return {
        split: _predict_single(
            dset,
            trainer=trainer,
            cache_dir=cache_dir,
            model=model,
            collate_fn=collate_fn,
            model_output_key=model_output_key,
            loader_kwargs=loader_kwargs,
            ts_kwargs=ts_kwargs,
            validate_store=validate_store,
            read_only=read_only,
        )
        for split, dset in sorted(dataset.items(), key=lambda x: x[0])
    }


def _predict_single(
    dataset: DatasetProtocol | datasets.Dataset,
    *,
    trainer: pl.Trainer,
    cache_dir: str | Path,
    model: torch.nn.Module | pl.LightningModule,
    collate_fn: pipes.Collate,
    model_output_key: Optional[str] = None,
    loader_kwargs: Optional[dict[str, Any] | DictConfig | loader_config.DataLoaderConfig] = None,
    ts_kwargs: Optional[dict[str, Any]] = None,
    validate_store: bool | int = True,
    read_only: bool = False,
) -> TensorStoreFactory:
    """Compute predictions for a dataset and store them in a tensorstore."""
    # set the fingerprint and define the store path
    op_fingerprint = _make_fingerprint(
        dataset=dataset,
        collate_fn=collate_fn,
        model=model,
        model_output_key=model_output_key,
    )
    logger.debug(f"Computing vectors with fingerprint `{op_fingerprint}`")
    index_path = Path(cache_dir, "predictions", f"{op_fingerprint}.ts")
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

    if read_only:
        if not index_path.exists():
            raise FileNotFoundError(f"Could not find TensorStore at path `{index_path}`.")
        logger.info(f"Loading TensorStore from path `{index_path}`. Shape={dset_shape}")
        store = ts_config.open(create=False)
    else:
        logger.debug(f"creating TensorStore at path `{index_path}`. Shape={dset_shape}")
        store = ts_config.open(create=not index_path.exists(), delete_existing=False)

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
            logger.warning("Prediction step was keyboard-interrupted. Deleting store.")
            shutil.rmtree(index_path)
        except Exception as e:
            del store
            raise e

    # wait for all workers to finish
    trainer.strategy.barrier(f"predict(fingerprint={op_fingerprint})")

    # validate that all store values have been initialized
    if validate_store:
        n_samples = math.inf if isinstance(validate_store, bool) else validate_store
        logger.debug(f"Validating the store using `{n_samples}` samples (all rows must be non-zero)")
        zero_ids = list(_get_zero_vec_indices(store, n_samples=n_samples))
        if len(zero_ids) > 0:
            max_display = 5
            frac = len(zero_ids) / len(dataset)
            zero_ids_ = zero_ids if len(zero_ids) < max_display else [str(x) for x in zero_ids[:max_display]] + ["..."]
            raise ValueError(
                f"Vector at indices {zero_ids_} are all zeros ({frac:.1%}). "
                f"This happens if the store has been initialized but not updated with predictions. "
                f"Please delete the store at `{index_path}` and try again. "
                f"NB: this could happen if the model outputs zero vectors."
            )

    # close the store and return the config
    del store
    return ts_config


def _infer_vector_shape(
    model: torch.nn.Module | pl.LightningModule,
    model_output_key: Optional[str],
    *,
    dataset: DatasetProtocol,
    collate_fn: pipes.Collate,
) -> tuple[int, ...]:
    # todo: handle variable length inputs # pylint: disable=fixme
    try:
        vector_shape = model.get_output_shape(model_output_key)
    except AttributeError as exc:
        logger.debug(
            f"{exc}. "
            f"Inferring the vector size by running one example through the model. "
            f"Implement `model.get_output_shape(output_key: str) -> tuple[int,...]` to skip this step."
        )
        batch = collate_fn([dataset[0]])
        one_vec = model.predict(batch) if hasattr(model, "predict") else model(batch)
        if model_output_key is not None:
            one_vec = one_vec[model_output_key]
        vector_shape = one_vec.shape[1:]

    return vector_shape


def _get_zero_vec_indices(store: tensorstore.TensorStore, n_samples: int) -> Iterable[int]:
    store_size = store.shape[0]
    ids = np.random.choice(store_size, n_samples, replace=False) if n_samples < store_size else range(store_size)
    prefetched, i = None, 0
    for i in track(ids, description="Validating store"):
        if prefetched is not None:
            vec = prefetched.result()
            if np.all(vec == 0):
                yield i - 1
        prefetched = store[i].read()

    if prefetched is not None:
        vec = prefetched.result()
        if np.all(vec == 0):
            yield i


@torch.inference_mode()
def _compute_and_store_predictions(
    trainer: pl.Trainer,
    dataset: DatasetProtocol,
    model: torch.nn.Module | pl.LightningModule,
    collate_fn: pipes.Collate,
    store: tensorstore.TensorStore,
    loader_kwargs: Optional[dict[str, Any] | DictConfig | loader_config.DataLoaderConfig] = None,
    model_output_key: Optional[str] = None,
) -> tensorstore.TensorStore:
    """Compute predictions for a dataset and store them in a tensorstore."""
    dset_with_ids: DatasetProtocol = DatasetWithIndices(dataset)
    collate_fn_with_ids: pipes.Collate = CollateWithIndices(collate_fn)
    pl_model: pl.LightningModule = _warp_as_lightning_model(model)

    # build the dataloader
    if isinstance(loader_kwargs, DictConfig):
        loader_kwargs = omegaconf.OmegaConf.to_container(loader_kwargs, resolve=True)
    if not isinstance(loader_kwargs, DataLoaderForPredictKwargs):
        if isinstance(loader_kwargs, pydantic.BaseModel):
            loader_kwargs = loader_kwargs.dict()
        loader_kwargs = loader_kwargs or {}
        if len(loader_kwargs) == 0:
            loader_kwargs = {"batch_size": 10}
            logger.warning("No `loader_kwargs` were provided. Using default batch_size=10. ")
        loader_kwargs = DataLoaderForPredictKwargs(**loader_kwargs)

    loader = torch.utils.data.DataLoader(dset_with_ids, collate_fn=collate_fn_with_ids, **loader_kwargs.dict())

    # process the dataset and store the predictions in the tensorstore
    with StorePredictions(trainer, store, model_output_key=model_output_key):
        trainer.predict(pl_model, dataloaders=loader, return_predictions=False)

    return store


_N_MAX_FGN_SAMPLES = 1000


def _get_dset_fingerprint(dataset: DatasetProtocol) -> str:
    if isinstance(dataset, datasets.Dataset):
        return dataset._fingerprint

    hasher = fingerprint.Hasher()
    hasher.update({"length": len(dataset)})

    # select random row ids
    if len(dataset) > _N_MAX_FGN_SAMPLES:
        rgn = np.random.RandomState(0)
        ids = rgn.choice(len(dataset), size=_N_MAX_FGN_SAMPLES, replace=False)
    else:
        ids = range(len(dataset))

    # hash the rows
    for i in ids:
        hasher.update(dataset[i])

    return hasher.hexdigest()


def _make_fingerprint(
    *,
    dataset: DatasetProtocol,
    collate_fn: pipes.Collate,
    model: torch.nn.Module | pl.LightningModule,
    model_output_key: Optional[str] = None,
) -> str:
    dset_fingerprint = _get_dset_fingerprint(dataset)
    model_fingerprint = _get_model_fingerprint(model)
    collate_fn_fingerprint = _get_collate_fn_fingerprint(collate_fn)
    op_fingerprint = f"{dset_fingerprint}_{model_fingerprint}_{collate_fn_fingerprint}"
    if model_output_key:
        op_fingerprint += f"_{model_output_key}"
    return op_fingerprint


def _get_model_fingerprint(model: torch.nn.Module) -> str:
    state = model.state_dict()
    hasher = fingerprint.Hasher()
    hasher.update(type(model).__name__)
    for k, v in sorted(state.items(), key=lambda x: x[0]):
        hasher.update(k)
        u = serialize_tensor(v)
        hasher.update(u)
    return hasher.hexdigest()


def _get_collate_fn_fingerprint(collate_fn: pipes.Collate) -> str:
    return fingerprint.Hasher.hash(collate_fn)
