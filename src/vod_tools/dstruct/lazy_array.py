# pylint: disable=no-member
from __future__ import annotations

import abc
from typing import Iterable, Optional

import datasets
import numpy as np
import tensorstore as ts

from src.vod_tools.dstruct.sized_dataset import SizedDataset, SliceType
from src.vod_tools.dstruct.ts_factory import TensorStoreFactory


class LazyArray(abc.ABC, SizedDataset[np.ndarray]):
    """A class that handles input array and provides lazy slicing into np.ndarray."""

    @abc.abstractmethod
    def __getitem__(self, item: SliceType) -> np.ndarray:
        """Slice the vector and return the result."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        return self._get_shape()

    def __len__(self) -> int:
        """Return the length of the vector."""
        return self.shape[0]

    def iter_batches(self, batch_size: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Iterate over vector elements."""
        for i in range(0, len(self), batch_size):
            j = min(i + batch_size, len(self))
            vec = self[i:j]
            if not isinstance(vec, np.ndarray):
                raise TypeError(f"Cannot handle type {type(vec)}")

            ids = np.arange(i, j)
            yield ids, vec

    def __repr__(self) -> str:
        """String representation of the vector handler."""
        return f"{type(self).__name__}(shape={self.shape})"


class ImplicitLazyArray(LazyArray):
    """Handles `SizedDataset`."""

    def __init__(self, data: SizedDataset[np.ndarray]):
        if not isinstance(data[0], np.ndarray):
            raise TypeError(f"Cannot handle type {type(data[0])}")
        self.data = data

    def __getitem__(self, item: SliceType) -> np.ndarray:
        """Slice the array and return the result."""
        return self.data[item]

    def _get_shape(self) -> tuple[int, ...]:
        return (len(self.data), *self[0].shape)


class TensorStoreLazyArray(LazyArray):
    """Handles tensorstore."""

    def __init__(self, store: ts.TensorStore):
        self.store = store

    def __getitem__(self, item: SliceType) -> np.ndarray:
        """Slice the stored vector and return the result."""
        if isinstance(item, slice) and item.stop is not None and item.stop > len(self):
            item = slice(item.start, self.store.shape[0], item.step)
        return self.store[item].read().result()

    def _get_shape(self) -> tuple[int, ...]:
        return self.store.shape


class TensorStoreFactoryLazyArray(TensorStoreLazyArray):
    """Handles TensorStoreFactory."""

    _open_store: Optional[ts.TensorStore] = None

    def __init__(self, factory: TensorStoreFactory):
        self.factory = factory

    @property
    def store(self) -> ts.TensorStore:
        """Return an open store."""
        if self._open_store is None:
            self._open_store = self.factory.open(create=False)
        return self._open_store

    def __getstate__(self) -> dict[str, object]:
        """Return the state of the object."""
        return {"factory": self.factory}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Set the state of the object."""
        self.factory = state["factory"]
        self._open_store = None


def as_lazy_array(
    x: SizedDataset[np.ndarray] | TensorStoreFactory | ts.TensorStore | np.ndarray,
) -> LazyArray:
    """Return a vector handler for the given vector type."""
    if isinstance(x, SizedDataset):
        return ImplicitLazyArray(x)

    if isinstance(x, TensorStoreFactory):
        return TensorStoreFactoryLazyArray(x)

    if isinstance(x, ts.TensorStore):
        return TensorStoreLazyArray(x)

    raise TypeError(f"Unsupported input type: {type(x)}")


@datasets.fingerprint.hashregister(ImplicitLazyArray)
def _hash_implicit_lazy_array(hasher: datasets.fingerprint.Hasher, obj: ImplicitLazyArray) -> str:
    return hasher.hash(obj.data)


@datasets.fingerprint.hashregister(TensorStoreLazyArray)
def _hash_store_lazy_array(hasher: datasets.fingerprint.Hasher, obj: TensorStoreLazyArray) -> str:
    return hasher.hash(obj.store)


@datasets.fingerprint.hashregister(TensorStoreFactoryLazyArray)
def _hash_store_factory_lazy_array(hasher: datasets.fingerprint.Hasher, obj: TensorStoreFactoryLazyArray) -> str:
    return hasher.hash(obj.factory)
