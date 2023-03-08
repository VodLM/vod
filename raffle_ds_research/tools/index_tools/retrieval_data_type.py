from __future__ import annotations

import abc
from abc import ABC
from enum import Enum
from numbers import Number
from typing import TypeVar, Union, Type, Generic, Iterable

import numpy as np
import torch


class RetrievalDataType(Enum):
    NUMPY = "NUMPY"
    TORCH = "TORCH"


Ts = TypeVar("Ts", bound=Union[np.ndarray, torch.Tensor])
Ts_o = TypeVar("Ts_o", bound=Union[np.ndarray, torch.Tensor])


def _type_repr(x: Ts) -> str:
    return f"{type(x).__name__}"


def _array_repr(x: Ts) -> str:
    return f"{type(x).__name__}(shape={x.shape}, dtype={x.dtype}))"


def _convert_array(scores: Union[np.ndarray, torch.Tensor], target_type: Type[Ts_o]) -> Ts_o:
    conversions = {
        torch.Tensor: {
            np.ndarray: torch.from_numpy,
            torch.Tensor: lambda x: x,
        },
        np.ndarray: {
            np.ndarray: lambda x: x,
            torch.Tensor: lambda x: x.detach().cpu().numpy(),
        },
    }
    converter = conversions[target_type][type(scores)]
    return converter(scores)


class RetrievalData(ABC, Generic[Ts]):
    __slots__ = ("scores", "indices")
    _expected_dim: int
    _str_sep: str = ""
    _repr_sep: str = ""
    scores: Ts
    indices: Ts

    def __init__(self, scores: Ts, indices: Ts):
        if scores.shape != indices.shape:
            raise ValueError(
                f"Scores and indices must have the same shape, "
                f"but got {_array_repr(self.scores)} and {_array_repr(self.indices)}"
            )
        if len(scores.shape) != self._expected_dim:
            raise ValueError(
                f"Scores and indices must be {self._expected_dim}D, "
                f"but got {_array_repr(scores)} and {_array_repr(indices)}"
            )
        self.scores = scores
        self.indices = indices

    def to(self, target_type: Type[Ts_o]) -> RetrievalData[Ts_o]:
        return type(self)(
            scores=_convert_array(self.scores, target_type),
            indices=_convert_array(self.indices, target_type),
        )

    @abc.abstractmethod
    def __getitem__(self, item) -> "RetrievalData":
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterable["RetrievalData"]:
        ...

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.scores.shape

    def _get_repr_parts(self) -> list[str]:
        parts = [
            f"{type(self).__name__}[{_type_repr(self.scores)}](",
            f"scores={repr(self.scores)}, ",
            f"indices={repr(self.indices)}",
            f")",
        ]

        return parts

    def __repr__(self) -> str:
        parts = self._get_repr_parts()
        return self._repr_sep.join(parts[:-1]) + parts[-1]

    def __str__(self) -> str:
        parts = self._get_repr_parts()
        return self._str_sep.join(parts[:-1]) + parts[-1]

    def __eq__(self, other: "RetrievalData") -> bool:
        op = {
            torch.Tensor: torch.all,
            np.ndarray: np.all,
        }[type(self.scores)]
        return op(self.scores == other.scores) and op(self.indices == other.indices)

    def to_dict(self) -> dict[str, list[Number]]:
        numpy_self = self.to(np.ndarray)
        return {
            "scores": self.scores.tolist(),
            "indices": self.indices.tolist(),
        }


class RetrievalTuple(RetrievalData[Ts]):
    _expected_dim = 0

    def __getitem__(self, item: str) -> RetrievalTuple[Ts]:
        return {
            "indices": self.indices,
            "scores": self.scores,
        }[item]

    def __iter__(self):
        raise NotImplementedError("RetrievalTuple is not iterable")


class RetrievalSample(RetrievalData[Ts]):
    _expected_dim = 1
    _str_sep: str = ""

    def __getitem__(self, item: int) -> RetrievalTuple[Ts]:
        return RetrievalTuple(
            scores=self.scores[item],
            indices=self.indices[item],
        )

    def __iter__(self) -> Iterable[RetrievalTuple[Ts]]:
        for i in range(len(self)):
            yield self[i]


class RetrievalBatch(RetrievalData[Ts]):
    _expected_dim = 2
    _str_sep: str = "\n"

    def __getitem__(self, item: int) -> RetrievalSample[Ts]:
        return RetrievalSample(
            scores=self.scores[item],
            indices=self.indices[item],
        )

    def __iter__(self) -> Iterable[RetrievalSample[Ts]]:
        for i in range(len(self)):
            yield self[i]
