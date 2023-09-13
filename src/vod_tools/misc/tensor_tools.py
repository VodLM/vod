import io
from abc import ABC, abstractmethod
from enum import Enum
from numbers import Number
from typing import Generic, Type, TypeVar, Union

import numpy as np
import torch

T = TypeVar("T")


class TensorFormatType(Enum):
    """Enum for the type of the tensor."""

    NUMPY = "numpy"
    TORCH = "torch"
    TENSORSTORE = "tensorstore"


class TensorFormat(object):
    """A tensor format."""

    format: TensorFormatType
    dtype: np.dtype
    device: torch.device


PyTensorType = Union[Number, list["PyTensorType"]]
TensorType = Union[np.ndarray, torch.Tensor, PyTensorType]

Ts = TypeVar("Ts", bound=TensorType)
AT = TypeVar("AT", bound="AbstractTensor")


class AbstractTensor(Generic[Ts], ABC):
    """An abstract tensor format for tensors/arrays (e.g., np.ndarray, torch.Tensor)."""

    _format: TensorFormat

    @property
    def format(self) -> TensorFormat:
        """Return the tensor format."""
        return self._format

    @abstractmethod
    def __getitem__(self, item: int | slice | tuple[int]) -> Ts:
        """Slice the tensor."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the tensor (first dim)."""
        ...

    @abstractmethod
    def reshape(self: AT, shape: tuple[int, ...]) -> AT:
        """Reshape the tensor."""
        ...

    @abstractmethod
    def to(self, format: Type[AT]) -> AT:
        """Cast the tensor to a different format."""
        ...

    @classmethod
    @abstractmethod
    def from_tensor(cls: Type[T], tensor: Ts | "AbstractTensor") -> T:
        """Create an `AbstractTensor` from a tensor."""
        ...


def serialize_tensor(x: torch.Tensor | np.ndarray) -> bytes:
    """Convert a torch.Tensor into a bytes object."""
    buff = io.BytesIO()
    if isinstance(x, torch.Tensor):
        if x.is_sparse:
            x = x.to_dense()
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float16)
        x = x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

    np.savez(buff, x)
    buff.seek(0)
    return buff.read()
