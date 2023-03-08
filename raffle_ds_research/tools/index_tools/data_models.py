from __future__ import annotations

from enum import Enum
from typing import Optional, TypeVar, Union, Generic

import numpy as np
import torch
from pydantic import BaseModel


class IoArrayType(Enum):
    NUMPY = "NUMPY"
    TORCH = "TORCH"


Ts = TypeVar("Ts", bound=Union[np.ndarray, torch.Tensor])


def type_repr(x: Ts) -> str:
    return f"{type(x).__name__}"


def array_repr(x: Ts) -> str:
    return f"{type(x).__name__}(shape={x.shape}, dtype={x.dtype}))"


class FaissResults(Generic[Ts]):
    scores: Ts
    indices: Ts

    def __init__(self, scores: Ts, indices: Ts):
        self.scores = scores
        self.indices = indices

    def _get_repr_parts(self) -> list[str]:
        parts = [
            f"{type(self).__name__}[{type_repr(self.scores)}](",
            f"scores={repr(self.scores)}, ",
            f"indices={repr(self.indices)},",
            f")",
        ]

        return parts

    def __repr__(self) -> str:
        return "".join(self._get_repr_parts())

    def __str__(self) -> str:
        return "\n".join(self._get_repr_parts())

    def __eq__(self, other: "FaissResults") -> bool:
        op = {
            torch.Tensor: torch.all,
            np.ndarray: np.all,
        }[type(self.scores)]
        return op(self.scores == other.scores) and op(self.indices == other.indices)


class FaissInitConfig(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

    index_path: str
    nprobe: int = 8


class InitResponse(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

    success: bool
    exception: Optional[str]


class SearchFaissQuery(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

    vectors: list  # todo [list[float]]
    top_k: int = 3


class FastSearchFaissQuery(BaseModel):
    """
    This is the same as SearchFaissQuery, but with the vectors serialized.
    todo: use protobuf
    """

    class Config:
        allow_mutation = False
        extra = "forbid"

    vectors: str
    array_type: IoArrayType = IoArrayType.NUMPY
    top_k: int = 3


class FaissSearchResponse(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

    scores: list  # todo [list[float]]
    indices: list  # todo [list[int]]


class FastFaissSearchResponse(BaseModel):
    """
    This is the same as FaissSearchResponse, but with the vectors serialized.
    todo: use protobuf
    """

    class Config:
        allow_mutation = False
        extra = "forbid"

    scores: str
    indices: str
