import abc
import typing as typ

import datasets
import numpy as np
from tensorstore import _tensorstore as ts
from vod_tools.ts_factory.ts_factory import TensorStoreFactory
from vod_types.sequence import Sequence


def _slice_sequence_of_arrays(arr: Sequence[np.ndarray], indices: slice) -> np.ndarray:
    indices_list = range(*indices.indices(len(arr)))
    return np.stack([arr[i] for i in indices_list])


class LazyArray(abc.ABC, Sequence[np.ndarray]):
    """A class that handles input array and provides lazy slicing into np.ndarray."""

    @typ.overload
    def __getitem__(self, __it: int) -> np.ndarray:
        ...

    @typ.overload
    def __getitem__(self, __it: slice) -> np.ndarray:
        ...

    def __getitem__(self, item: int | slice) -> np.ndarray:
        """Slice the vector and return the result."""
        if isinstance(item, int):
            return self._getitem_int(item)
        if isinstance(item, slice):
            return self._getitem_slice(item)
        raise TypeError(f"Unsupported index type: {type(item)}")

    @abc.abstractmethod
    def _getitem_int(self, item: int) -> np.ndarray:
        """Slice the vector and return the result."""
        raise NotImplementedError

    def _getitem_slice(self, item: slice) -> np.ndarray:
        return self._slice_arr(item)

    @abc.abstractmethod
    def _get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _slice_arr(self, indices: slice) -> np.ndarray:
        """Slice the vector and return the result."""
        return _slice_sequence_of_arrays(self, indices)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        return self._get_shape()

    def __len__(self) -> int:
        """Return the length of the vector."""
        return self.shape[0]

    def __repr__(self) -> str:
        """String representation of the vector handler."""
        return f"{type(self).__name__}(shape={self.shape})"


class NumpyLazyArray(LazyArray):
    """Handles sequences of `np.ndarray`."""

    def __init__(self, data: Sequence[np.ndarray]):
        if not isinstance(data[0], np.ndarray):
            raise TypeError(f"Cannot handle type {type(data[0])}")
        self.data = data

    def _getitem_int(self, item: int) -> np.ndarray:
        """Slice the array and return the result."""
        return self.data[item]

    def _get_shape(self) -> tuple[int, ...]:
        return (len(self.data), *self[0].shape)

    def _slice_arr(self, indices: slice) -> np.ndarray:
        """Slice the vector and return the result."""
        if isinstance(self.data, np.ndarray):
            return self.data[indices]
        return _slice_sequence_of_arrays(self, indices)


class TensorStoreLazyArray(LazyArray):
    """Handles tensorstore."""

    def __init__(self, store: ts.TensorStore):
        self.store = store

    def _getitem_int(self, item: int) -> np.ndarray:
        """Slice the stored vector and return the result."""
        return self.store[item].read().result()

    def _get_shape(self) -> tuple[int, ...]:
        return self.store.shape

    def _slice_arr(self, indices: slice) -> np.ndarray:
        """Slice the vector and return the result."""
        start, stop, step = indices.indices(len(self))
        stop = min(stop, len(self))
        truncated_indices = slice(start, stop, step)
        return self.store[truncated_indices].read().result()


class TensorStoreFactoryLazyArray(TensorStoreLazyArray):
    """Handles TensorStoreFactory."""

    _open_store: None | ts.TensorStore = None

    def __init__(self, factory: TensorStoreFactory):
        self.factory = factory

    @property
    def store(self) -> ts.TensorStore:
        """Return an open store."""
        if self._open_store is None:
            self._open_store = self.factory.open(create=False)  # type: ignore
        return self._open_store

    def __getstate__(self) -> dict[str, object]:
        """Return the state of the object."""
        return {"factory": self.factory}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Set the state of the object."""
        self.factory = state["factory"]
        self._open_store = None


Array: typ.TypeAlias = Sequence[np.ndarray] | TensorStoreFactory | ts.TensorStore | np.ndarray


def as_lazy_array(x: Array) -> LazyArray:
    """Return a vector handler for the given vector type."""
    if isinstance(x, Sequence):
        return NumpyLazyArray(x)

    if isinstance(x, TensorStoreFactory):
        return TensorStoreFactoryLazyArray(x)

    if isinstance(x, ts.TensorStore):
        return TensorStoreLazyArray(x)

    raise TypeError(f"Unsupported input type: {type(x)}")


@datasets.fingerprint.hashregister(NumpyLazyArray)
def _hash_implicit_lazy_array(hasher: datasets.fingerprint.Hasher, obj: NumpyLazyArray) -> str:
    return hasher.hash(obj.data)


@datasets.fingerprint.hashregister(TensorStoreLazyArray)
def _hash_store_lazy_array(hasher: datasets.fingerprint.Hasher, obj: TensorStoreLazyArray) -> str:
    return hasher.hash(obj.store)


@datasets.fingerprint.hashregister(TensorStoreFactoryLazyArray)
def _hash_store_factory_lazy_array(hasher: datasets.fingerprint.Hasher, obj: TensorStoreFactoryLazyArray) -> str:
    return hasher.hash(obj.factory)


def slice_arrays_sequence(arr: Sequence[np.ndarray], indices: slice) -> np.ndarray:
    """Slice an array and return the result."""
    if isinstance(arr, np.ndarray):
        return arr[indices]
    if isinstance(arr, LazyArray):
        return arr._slice_arr(indices)

    return _slice_sequence_of_arrays(arr, indices)
