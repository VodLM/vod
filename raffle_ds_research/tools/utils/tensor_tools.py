from __future__ import annotations

import io
from abc import ABC, abstractmethod
from enum import Enum
from numbers import Number
from typing import Generic, Type, TypeVar, Union

import numpy as np
import torch
from typing_extensions import Self


class TensorFormatType(Enum):
    NUMPY = "numpy"
    TORCH = "torch"
    TENSORSTORE = "tensorstore"


class TensorFormat(object):
    format: TensorFormatType
    dtype: np.dtype
    device: torch.device


PyTensorType = Union[Number, list["PyTensorType"]]
TensorType = Union[np.ndarray, torch.Tensor, PyTensorType]

Ts = TypeVar("Ts", bound=TensorType)


class AbstractTensor(Generic[Ts], ABC):
    _format: TensorFormat

    @property
    def format(self) -> TensorFormat:
        return self._format

    @abstractmethod
    def __getitem__(self, item: int | slice | tuple[int]) -> Ts:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def reshape(self, shape: tuple[int, ...]) -> Self:
        ...

    @abstractmethod
    def to(self, format: Type[AbstractTensor]) -> Self:
        ...

    @classmethod
    @abstractmethod
    def from_tensor(cls, tensor: Ts | "AbstractTensor") -> Self:
        ...


def serialize_tensor(x: torch.Tensor | np.ndarray) -> bytes:
    """Convert a torch.Tensor into a bytes object."""
    buff = io.BytesIO()
    if isinstance(x, torch.Tensor):
        if x.is_sparse:
            x = x.to_dense()
        x = x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

    np.savez(buff, x)
    buff.seek(0)
    return buff.read()
