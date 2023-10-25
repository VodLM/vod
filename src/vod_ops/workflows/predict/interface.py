import contextlib  # noqa: I001
import pathlib
import shutil
import typing as typ
from pathlib import Path

import lightning as L
import numpy as np
from tensorstore import _tensorstore as ts
import torch
from loguru import logger
from omegaconf import DictConfig
from rich.progress import track
from torch.utils.data.dataloader import default_collate
import vod_configs
from vod_tools.ts_factory.ts_factory import TensorStoreFactory
import vod_types as vt

from .compute import LoaderKwargs, compute_and_store_predictions
from .fingerprint import make_predict_fingerprint


class StoreValidationError(ValueError):
    """Raised when a store is not valid."""


class Predict:
    """Compute vectors for a dataset and store them in a tensorstore."""

    __slots__ = (
        "save_dir",
        "_dataset",
        "_model",
        "_collate_fn",
        "_model_output_key",
        "_fingerprint",
    )

    def __init__(
        self,
        *,
        dataset: vt.Sequence,
        save_dir: str | pathlib.Path,
        model: torch.nn.Module | vt.EncoderLike,
        collate_fn: vt.Collate = default_collate,  # type: ignore
        model_output_key: None | str = None,
    ):
        self._dataset = dataset
        self._model = model
        self._collate_fn = collate_fn
        self._model_output_key = model_output_key
        self.save_dir = pathlib.Path(save_dir).expanduser()

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
            f"cache_dir={self.save_dir})"
        )

    def __call__(
        self,
        fabric: None | L.Fabric = None,
        loader_kwargs: None | LoaderKwargs = None,
        ts_kwargs: None | dict[str, typ.Any] = None,
        validate_store: bool | int = True,
        open_mode: None | typ.Literal["x", "r", "a"] = None,
    ) -> TensorStoreFactory:
        """Compute vectors for a dataset and store them in a tensorstore."""
        if open_mode in {None, "r"} and self.exists():
            logger.info(f"Store already exists at `{self.store_path}`.")
            store_factory = self.read()
            store = store_factory.open(create=False)
            if not validate_store or self.validate_store(n_samples=validate_store):
                return store_factory
        if open_mode in {"r"}:
            raise FileNotFoundError(f"Store does not exist at `{self.store_path}` or is invalid.")
        if open_mode in {"a"} and not self.exists():
            raise FileNotFoundError(f"Store does not exist at `{self.store_path}`.")

        # create or open the store
        if open_mode in {None, "x"}:
            store = self.instantiate(ts_kwargs).open(create=False)
        elif open_mode in {"a"}:
            store = self.read().open(create=False)
        else:
            raise ValueError(f"Invalid `open_mode`: {open_mode}")

        # compute the vectors
        self._compute_vectors(store, fabric=fabric, loader_kwargs=(loader_kwargs or {}))

        # validate the store
        if validate_store:
            self.validate_store(n_samples=validate_store)

        return self.read()

    def _compute_vectors(
        self,
        store: ts.TensorStore,
        fabric: None | L.Fabric,
        loader_kwargs: LoaderKwargs,
    ) -> None:
        if fabric is None:
            fabric = L.Fabric()
            fabric.launch()
        try:
            fabric.strategy.barrier(f"predict-start(fingerprint={self.fingerprint})")
            compute_and_store_predictions(
                fabric=fabric,
                dataset=self._dataset,
                model=self._model,
                model_output_key=self._model_output_key,
                collate_fn=self._collate_fn,
                store=store,
                loader_kwargs=loader_kwargs,
            )
            fabric.strategy.barrier(f"predict-ends(fingerprint={self.fingerprint})")
        except KeyboardInterrupt as exc:
            logger.warning(f"`Predict` was keyboard-interrupted. Deleting store at `{self.store_path}`.")
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(self.store_path)
            raise exc
        except Exception as exc:
            logger.warning(
                f"`Predict` was failed with exception `{type(exc).__name__}`. Deleting store at `{self.store_path}`."
            )
            shutil.rmtree(self.store_path)
            raise exc

    def instantiate(self, ts_kwargs: None | dict[str, typ.Any] = None) -> TensorStoreFactory:
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
        factory = TensorStoreFactory.instantiate(
            path=self.store_path,
            shape=dset_shape,
            **(ts_kwargs or {}),
        )

        factory.open(create=True, delete_existing=False)
        return factory

    def read(self) -> TensorStoreFactory:
        """Open the tensorstore in read mode."""
        return TensorStoreFactory.from_path(path=self.store_path)

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
        return _predict_store_path(cache_dir=self.save_dir, op_fingerprint=self.fingerprint)

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
        store = self.read().open()
        n_samples = len(self._dataset) if isinstance(n_samples, bool) else n_samples
        logger.debug(f"Validating store at `{self.store_path}` with {n_samples} samples.")
        zero_ids = list(_get_zero_vec_indices(store, n_samples=n_samples))
        if raise_exc and len(zero_ids) > 0:
            frac = len(zero_ids) / len(self._dataset)
            zero_ids_ = zero_ids if len(zero_ids) < max_display else [str(x) for x in zero_ids[:max_display]] + ["..."]
            raise StoreValidationError(
                f"Vector at indices {zero_ids_} are all zeros ({frac:.1%}). "
                f"This happens if the store has been initialized but not updated with predictions. "
                f"Please delete the store at `{self.store_path}` and try again. "
                f"NOTE: this could happen if the model outputs zero vectors."
            )

        return len(zero_ids) == 0


def predict(  # noqa: PLR0913
    dataset: vt.Sequence,
    *,
    fabric: L.Fabric,
    cache_dir: str | Path,
    model: torch.nn.Module | vt.EncoderLike,
    collate_fn: vt.Collate,
    model_output_key: None | str = None,
    loader_kwargs: None | dict[str, typ.Any] | DictConfig | vod_configs.DataLoaderConfig = None,
    ts_kwargs: None | dict[str, typ.Any] = None,
    validate_store: bool | int = True,
    open_mode: None | typ.Literal["x", "r", "a"] = None,
) -> TensorStoreFactory:
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
        save_dir=cache_dir,
    )
    return predict_fn(
        fabric=fabric,
        loader_kwargs=loader_kwargs,
        ts_kwargs=ts_kwargs,
        validate_store=validate_store,
        open_mode=open_mode,
    )


def _infer_vector_shape(
    model: torch.nn.Module | vt.EncoderLike,
    model_output_key: None | str,
    *,
    dataset: vt.Sequence,
    collate_fn: vt.Collate,
) -> tuple[int, ...]:
    try:
        vector_shape = model.get_encoding_shape()  # type: ignore
    except AttributeError as exc:
        logger.debug(
            f"{exc}. "
            f"Inferring the vector size by running one example through the model. "
            f"Implement `model.get_encoding_shape(output_key: str) -> tuple[int,...]` to skip this step."
        )
        batch = collate_fn([dataset[0]])
        one_vec = model(batch)  # type: ignore
        if model_output_key is not None:
            one_vec = one_vec[model_output_key]
        vector_shape = one_vec.shape[1:]  # type: ignore

    return vector_shape


def _get_zero_vec_indices(store: ts.TensorStore, n_samples: int) -> typ.Iterable[int]:
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


def _predict_store_path(*, op_fingerprint: str, cache_dir: pathlib.Path | str) -> pathlib.Path:
    return pathlib.Path(cache_dir, "predictions", f"{op_fingerprint}.ts")
