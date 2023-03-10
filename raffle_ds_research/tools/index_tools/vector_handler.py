from __future__ import annotations

import abc
from typing import Iterable, TypeVar, Union

import numpy as np
import rich
import tensorstore
import torch
from typing_extensions import TypeAlias

from raffle_ds_research.tools import TensorStoreFactory

VectorType: TypeAlias = Union[torch.Tensor, tensorstore.TensorStore, np.ndarray, TensorStoreFactory]
Ts = TypeVar("Ts", bound=Union[torch.Tensor, np.ndarray])


class VectorHandler(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, item: int | slice | Iterable[int]) -> Ts:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        return self._get_shape()

    @abc.abstractmethod
    def _get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def __len__(self):
        return self.shape[0]

    def iter_batches(self, batch_size: int) -> Iterable[tuple[Ts, Ts]]:
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

    def __repr__(self):
        return f"{type(self).__name__}(shape={self.shape})"


class TorchVectorHandler(VectorHandler):
    def __init__(self, vectors: torch.Tensor):
        self.vectors = vectors.detach().data.cpu()

    def __getitem__(self, item: int | slice | Iterable[int]) -> torch.Tensor:
        return self.vectors[item]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.vectors.shape


class NumpyVectorHandler(VectorHandler):
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def __getitem__(self, item: int | slice | Iterable[int]) -> np.ndarray:
        return self.vectors[item]

    def _get_shape(self) -> tuple[int, ...]:
        return self.vectors.shape


class TensorStoreVectorHandler(VectorHandler):
    def __init__(self, vectors: tensorstore.TensorStore):
        self.vectors = vectors

    def __getitem__(self, item: int | slice | Iterable[int]) -> np.ndarray:
        return self.vectors[item].read().result()

    def _get_shape(self) -> tuple[int, ...]:
        return self.vectors.shape


def vector_handler(vectors: VectorType) -> VectorHandler:
    if isinstance(vectors, torch.Tensor):
        return TorchVectorHandler(vectors)
    elif isinstance(vectors, TensorStoreFactory):
        vectors = vectors.open(create=False)
        return vector_handler(vectors)
    elif isinstance(vectors, tensorstore.TensorStore):
        return TensorStoreVectorHandler(vectors)
    elif isinstance(vectors, np.ndarray):
        return NumpyVectorHandler(vectors)
    else:
        raise TypeError(f"Unsupported vector type: {type(vectors)}")
