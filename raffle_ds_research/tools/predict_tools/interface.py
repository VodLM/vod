from __future__ import annotations  # noqa: I001

import math
import pathlib
import shutil
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union
from typing_extensions import TypeAlias

import datasets
import lightning.pytorch as pl
import numpy as np
import pydantic
import torch
from datasets import fingerprint
from loguru import logger
from omegaconf import DictConfig, omegaconf
from rich.progress import track
import tensorstore as ts
from torch.utils import data as torch_data
from torch.utils.data.dataloader import default_collate

from raffle_ds_research.tools import pipes, dstruct
from raffle_ds_research.tools.utils import loader_config
from raffle_ds_research.tools.utils.tensor_tools import serialize_tensor

from .callback import StorePredictions
from .wrappers import CollateWithIndices, DatasetWithIndices, _warp_as_lightning_model


LoaderKwargs: TypeAlias = Union[dict[str, Any], DictConfig, loader_config.DataLoaderConfig]


class DataLoaderForPredictKwargs(loader_config.DataLoaderConfig):
    """Confiuguration for `torch.utils.data.Dataloader` for predictions."""

    @pydantic.validator("shuffle", pre=True)
    def _force_shuffle(cls, value: bool) -> bool:
        if value:
            logger.debug("Shuffle is set to True. This is unnecessary for predictions. Forcing `shuffle` to False.")
        return False


class Predict:
    """Compute vectors for a dataset and store them in a tensorstore."""

    __slots__ = (
        "cache_dir",
        "_dataset",
        "_model",
        "_collate_fn",
        "_model_output_key",
        "_fingerprint",
    )

    def __init__(
        self,
        *,
        dataset: dstruct.SizedDataset,
        cache_dir: str | pathlib.Path,
        model: Union[torch.nn.Module, pl.LightningModule],
        collate_fn: pipes.Collate = default_collate,  # type: ignore
        model_output_key: Optional[str] = None,
    ):
        self._dataset = dataset
        self._model = model
        self._collate_fn = collate_fn
        self._model_output_key = model_output_key
        self.cache_dir = pathlib.Path(cache_dir).expanduser()

        # compute the fingerprint of the resulting function
        self._fingerprint = make_predict_fingerprint(
            dataset=self._dataset,
            model=self._model,
            collate_fn=self._collate_fn,
            model_output_key=self._model_output_key,
        )

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return (
            f"{self.__class__.__name__}("
            f"dataset={type(self._dataset).__name__}, "
            f"model={type(self._model).__name__}, "
            f"collate_fn={type(self._collate_fn).__name__}, "
            f"model_output_key={self._model_output_key}, "
            f"cache_dir={self.cache_dir})"
        )

    def __call__(
        self,
        trainer: Optional[pl.Trainer] = None,
        loader_kwargs: Optional[LoaderKwargs] = None,
        ts_kwargs: Optional[dict[str, Any]] = None,
        validate_store: bool | int = True,
        open_mode: Optional[Literal["x", "r", "a"]] = None,
    ) -> dstruct.TensorStoreFactory:
        """Compute vectors for a dataset and store them in a tensorstore."""
        if open_mode in {None, "r"} and self.exists():
            logger.info(f"Store already exists at `{self.store_path}`.")
            store_factory = self.read()
            store = store_factory.open(create=False)
            if not validate_store or self.validate_store(store, n_samples=validate_store):
                return store_factory
        if open_mode in {"r"}:
            raise FileNotFoundError(f"Store does not exist at `{self.store_path}` or is invalid.")
        if open_mode in {"a"} and not self.exists():
            raise FileNotFoundError(f"Store does not exist at `{self.store_path}`.")

        # create or open the store
        if open_mode in {None, "x"}:
            store = self.instantiate(ts_kwargs).open(create=True)
        elif open_mode in {"a"}:
            store = self.read().open(create=False)
        else:
            raise ValueError(f"Invalid `open_mode`: {open_mode}")

        # compute the vectors
        self._compute_vectors(store, trainer=trainer, loader_kwargs=(loader_kwargs or {}))

        # validate the store
        if validate_store:
            self.validate_store(n_samples=validate_store)

        return self.read()

    def _compute_vectors(
        self,
        store: ts.TensorStore,
        trainer: Optional[pl.Trainer],
        loader_kwargs: LoaderKwargs,
    ) -> None:
        if trainer is None:
            trainer = pl.Trainer()
        try:
            trainer.strategy.barrier(f"predict-start(fingerprint={self.fingerprint})")
            _compute_and_store_predictions(
                trainer=trainer,
                dataset=self._dataset,
                model=self._model,
                model_output_key=self._model_output_key,
                collate_fn=self._collate_fn,
                store=store,
                loader_kwargs=loader_kwargs,
            )
            trainer.strategy.barrier(f"predict-ends(fingerprint={self.fingerprint})")
        except KeyboardInterrupt as exc:
            logger.warning(f"`Predict` was keyboard-interrupted. Deleting store at `{self.store_path}`.")
            shutil.rmtree(self.store_path)
            raise exc
        except Exception as exc:
            logger.warning(
                f"`Predict` was failed with exception `{type(exc).__name__}`. " "Deleting store at `{self.store_path}`."
            )
            shutil.rmtree(self.store_path)
            raise exc

    def instantiate(self, ts_kwargs: Optional[dict[str, Any]] = None) -> dstruct.TensorStoreFactory:
        """Create the tensorstore (write mode)."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if self.exists():
            raise FileExistsError(f"Store already exists at `{self.store_path}`.")

        # infer the vector dimensions
        vector_shape = _infer_vector_shape(
            self._model,
            model_output_key=self._model_output_key,
            dataset=self._dataset,
            collate_fn=self._collate_fn,
        )

        # build the store config and open the store
        dset_shape = (len(self._dataset), *vector_shape)
        factory = dstruct.TensorStoreFactory.instantiate(
            path=self.store_path,
            shape=dset_shape,
            **(ts_kwargs or {}),
        )

        factory.open(create=True)
        return factory

    def read(self) -> dstruct.TensorStoreFactory:
        """Open the tensorstore in read mode."""
        return dstruct.TensorStoreFactory.from_path(path=self.store_path)

    def exists(self) -> bool:
        """Return whether the store exists."""
        return self.store_path.exists()

    def rm(self) -> None:
        """Remove the store."""
        if self.exists():
            shutil.rmtree(self.store_path)

    @property
    def fingerprint(self) -> str:
        """Returns the fingerprint of the `Predict` function."""
        return self._fingerprint

    @property
    def store_path(self) -> pathlib.Path:
        """Returns the path to the store."""
        return _predict_store_path(cache_dir=self.cache_dir, op_fingerprint=self.fingerprint)

    def validate_store(
        self,
        n_samples: bool | int = True,
        max_display: int = 5,
        raise_exc: bool = True,
    ) -> bool:
        """Validate that the store is not empty."""
        if not self.exists():
            if raise_exc:
                raise FileNotFoundError(f"Store at `{self.store_path}` does not exist.")
            return False
        store = self.read().open(create=False)
        n_samples = len(self._dataset) if isinstance(n_samples, bool) else n_samples
        logger.info(f"Validating store at `{self.store_path}` with {n_samples} samples.")
        zero_ids = list(_get_zero_vec_indices(store, n_samples=n_samples))
        if raise_exc and len(zero_ids) > 0:
            frac = len(zero_ids) / len(self._dataset)
            zero_ids_ = zero_ids if len(zero_ids) < max_display else [str(x) for x in zero_ids[:max_display]] + ["..."]
            raise StoreValidationError(
                f"Vector at indices {zero_ids_} are all zeros ({frac:.1%}). "
                f"This happens if the store has been initialized but not updated with predictions. "
                f"Please delete the store at `{self.store_path}` and try again. "
                f"NB: this could happen if the model outputs zero vectors."
            )

        return len(zero_ids) == 0


def predict(
    dataset: dstruct.SizedDataset,
    *,
    trainer: pl.Trainer,
    cache_dir: str | Path,
    model: torch.nn.Module | pl.LightningModule,
    collate_fn: pipes.Collate,
    model_output_key: Optional[str] = None,
    loader_kwargs: Optional[dict[str, Any] | DictConfig | loader_config.DataLoaderConfig] = None,
    ts_kwargs: Optional[dict[str, Any]] = None,
    validate_store: bool | int = True,
    open_mode: Optional[Literal["x", "r", "a"]] = None,
) -> dstruct.TensorStoreFactory:
    """Compute predictions for a dataset and store them in a tensorstore.

    Open modes:
        None - read if existing, else write.
        "x" - write only, fail if exists
        "r" - read only, fail if not exists
        "a" - override existing store, fail if not exists

    """
    predict_fn = Predict(
        dataset=dataset,
        model=model,
        collate_fn=collate_fn,
        model_output_key=model_output_key,
        cache_dir=cache_dir,
    )
    return predict_fn(
        trainer=trainer,
        loader_kwargs=loader_kwargs,
        ts_kwargs=ts_kwargs,
        validate_store=validate_store,
        open_mode=open_mode,
    )


def _infer_vector_shape(
    model: torch.nn.Module | pl.LightningModule,
    model_output_key: Optional[str],
    *,
    dataset: dstruct.SizedDataset,
    collate_fn: pipes.Collate,
) -> tuple[int, ...]:
    try:
        vector_shape = model.get_output_shape(model_output_key)  # type: ignore
    except AttributeError as exc:
        logger.debug(
            f"{exc}. "
            f"Inferring the vector size by running one example through the model. "
            f"Implement `model.get_output_shape(output_key: str) -> tuple[int,...]` to skip this step."
        )
        batch = collate_fn([dataset[0]])
        one_vec = model.predict(batch) if hasattr(model, "predict") else model(batch)  # type: ignore
        if model_output_key is not None:
            one_vec = one_vec[model_output_key]
        vector_shape = one_vec.shape[1:]

    return vector_shape


def _get_zero_vec_indices(store: ts.TensorStore, n_samples: int) -> Iterable[int]:
    """Validate that the store has been written in all positions."""
    store_size = store.shape[0]
    if n_samples < store_size:
        n_samples_group = n_samples // 3
        # sample uniformly spaced indices
        uniform_ids = np.linspace(0, store_size - 1, n_samples_group, dtype=int)
        # sample consecutive indices from the end
        consecutive_ids = np.linspace(store_size - n_samples_group, store_size - 1, n_samples_group, dtype=int)
        # sample random indices among all ids
        random_ids = np.random.randint(0, store_size, size=(n_samples_group,))
        ids = np.concatenate([uniform_ids, consecutive_ids, random_ids])
    else:
        ids = range(store_size)
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


class StoreValidationError(ValueError):
    """Raised when a store is not valid."""


def _predict_store_path(*, op_fingerprint: str, cache_dir: pathlib.Path | str) -> pathlib.Path:
    return pathlib.Path(cache_dir, "predictions", f"{op_fingerprint}.ts")


@torch.inference_mode()
def _compute_and_store_predictions(
    trainer: pl.Trainer,
    dataset: dstruct.SizedDataset,
    model: torch.nn.Module | pl.LightningModule,
    collate_fn: pipes.Collate,
    store: ts.TensorStore,
    loader_kwargs: Optional[LoaderKwargs] = None,
    model_output_key: Optional[str] = None,
) -> ts.TensorStore:
    """Compute predictions for a dataset and store them in a tensorstore."""
    dset_with_ids: dstruct.SizedDataset[dict] = DatasetWithIndices[dict](dataset)
    collate_fn_with_ids: pipes.Collate = CollateWithIndices(collate_fn)
    pl_model: pl.LightningModule = _warp_as_lightning_model(model)

    # build the dataloader
    if isinstance(loader_kwargs, DictConfig):
        loader_kwargs = omegaconf.OmegaConf.to_container(loader_kwargs, resolve=True)  # type: ignore
    if not isinstance(loader_kwargs, DataLoaderForPredictKwargs):
        if isinstance(loader_kwargs, pydantic.BaseModel):
            loader_kwargs = loader_kwargs.dict()
        loader_kwargs = loader_kwargs or {}
        if len(loader_kwargs) == 0:
            loader_kwargs = {"batch_size": 10}
            logger.warning("No `loader_kwargs` were provided. Using default batch_size=10. ")
        loader_kwargs = DataLoaderForPredictKwargs(**loader_kwargs)  # type: ignore

    loader = torch_data.DataLoader(
        dset_with_ids,  # type: ignore
        collate_fn=collate_fn_with_ids,
        **loader_kwargs.dict(),
    )

    # process the dataset and store the predictions in the tensorstore
    with StorePredictions(trainer, store, model_output_key=model_output_key):
        trainer.predict(pl_model, dataloaders=loader, return_predictions=False)

    return store


def make_predict_fingerprint(
    *,
    dataset: dstruct.SizedDataset | datasets.Dataset,
    collate_fn: pipes.Collate,
    model: torch.nn.Module | pl.LightningModule,
    model_output_key: Optional[str] = None,
) -> str:
    """Make a fingerprint for the `predict` operation."""
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


def _get_dset_fingerprint(dataset: dstruct.SizedDataset | datasets.Dataset, max_samples: float = math.inf) -> str:
    if isinstance(dataset, datasets.Dataset):
        return dataset._fingerprint

    hasher = fingerprint.Hasher()
    hasher.update({"length": len(dataset)})

    # select random row ids
    if len(dataset) > max_samples:
        rgn = np.random.RandomState(0)
        ids = rgn.choice(len(dataset), size=int(max_samples), replace=False)
    else:
        ids = range(len(dataset))

    # hash the rows
    for i in ids:
        hasher.update(dataset[i])

    return hasher.hexdigest()
