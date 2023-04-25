# pylint: disable=no-member
from __future__ import annotations

import abc
from typing import Generic, Iterable, TypeVar, Union

import numpy as np
import tensorstore
import torch
from typing_extensions import TypeAlias

from raffle_ds_research.tools import TensorStoreFactory

VectorType: TypeAlias = Union[torch.Tensor, tensorstore.TensorStore, np.ndarray, TensorStoreFactory]
Ts = TypeVar("Ts", bound=Union[torch.Tensor, np.ndarray])


class VectorHandler(Generic[Ts], abc.ABC):
    """A class that handles vectors of type arbitraty type `Ts`."""

    @abc.abstractmethod
    def __getitem__(self, item: int | slice | Iterable[int]) -> Ts:
        """Slice the vector and return the result."""
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        return self._get_shape()

    @abc.abstractmethod
    def _get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the length of the vector."""
        return self.shape[0]

    def iter_batches(self, batch_size: int) -> Iterable[tuple[Ts, Ts]]:
        """Iterate over vector elements."""
        for i in range(0, len(self), batch_size):
            j = min(i + batch_size, len(self))
            vec = self[i:j]
            if isinstance(vec, torch.Tensor):
                ids = torch.arange(i, j, device=vec.device)
            elif isinstance(vec, np.ndarray):
                ids = np.arange(i, j)
            else:
                raise TypeError(f"Cannot handle type {type(vec)}")

            yield ids, vec

    def __repr__(self) -> str:
        """String representation of the vector handler."""
        return f"{type(self).__name__}(shape={self.shape})"


class TorchVectorHandler(VectorHandler[torch.Tensor]):
    """Handles torch vectors."""

    def __init__(self, vectors: torch.Tensor):
        self.vectors = vectors.detach().data.cpu()

    def __getitem__(self, item: int | slice | Iterable[int]) -> torch.Tensor:
        """Slice the tensor and return the result."""
        return self.vectors[item]

    def _get_shape(self) -> tuple[int, ...]:
        return self.vectors.shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        return self.vectors.shape


class NumpyVectorHandler(VectorHandler[np.ndarray]):
    """Handles numpy vectors."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def __getitem__(self, item: int | slice | Iterable[int]) -> np.ndarray:
        """Slice the array and return the result."""
        return self.vectors[item]

    def _get_shape(self) -> tuple[int, ...]:
        return self.vectors.shape


class TensorStoreVectorHandler(VectorHandler[np.ndarray]):
    """Handles tensorstore vectors."""

    def __init__(self, vectors: tensorstore.TensorStore):
        self.vectors = vectors

    def __getitem__(self, item: int | slice | Iterable[int]) -> np.ndarray:
        """Slice the stored vector and return the result."""
        return self.vectors[item].read().result()

    def _get_shape(self) -> tuple[int, ...]:
        return self.vectors.shape


def vector_handler(vectors: VectorType) -> VectorHandler:
    """Return a vector handler for the given vector type."""
    if isinstance(vectors, torch.Tensor):
        return TorchVectorHandler(vectors)

    if isinstance(vectors, TensorStoreFactory):
        vectors = vectors.open(create=False)
        return vector_handler(vectors)

    if isinstance(vectors, tensorstore.TensorStore):
        return TensorStoreVectorHandler(vectors)

    if isinstance(vectors, np.ndarray):
        return NumpyVectorHandler(vectors)

    raise TypeError(f"Unsupported vector type: {type(vectors)}")
