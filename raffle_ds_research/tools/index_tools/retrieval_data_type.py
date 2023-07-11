from __future__ import annotations

import abc
import copy
from abc import ABC
from enum import Enum
from numbers import Number
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar, Union

import numpy as np
import torch
from typing_extensions import Type, TypeAlias

from raffle_ds_research.tools import c_tools


class RetrievalDataType(Enum):
    """Type of retrieval data."""

    NUMPY = "NUMPY"
    TORCH = "TORCH"


SliceType: TypeAlias = Union[int, slice, Iterable[int]]
Ts = TypeVar("Ts", np.ndarray, torch.Tensor)
Ts_co = TypeVar("Ts_co", np.ndarray, torch.Tensor, covariant=True)
Ts_out = TypeVar("Ts_out", np.ndarray, torch.Tensor)


def _type_repr(x: Ts) -> str:
    return f"{type(x).__name__}"


def _array_repr(x: Ts) -> str:
    return f"{type(x).__name__}(shape={x.shape}, dtype={x.dtype}))"


def _convert_array(scores: Union[np.ndarray, torch.Tensor], target_type: Type[Ts_out]) -> Ts_out:
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


def _stack_arrays(arrays: Iterable[Ts]) -> Ts:
    first, *rest = arrays
    operator = {
        torch.Tensor: torch.stack,
        np.ndarray: np.stack,
    }[type(first)]
    return operator([first, *rest])


def _concat_arrays(arrays: Iterable[Ts]) -> Ts:
    first, *rest = arrays
    operator = {
        torch.Tensor: torch.cat,
        np.ndarray: np.concatenate,
    }[type(first)]
    return operator([first, *rest])


def _full_like(x: Ts, fill_value: Union[int, float]) -> Ts:
    return {
        torch.Tensor: torch.full_like,
        np.ndarray: np.full_like,
    }[
        type(x)
    ](x, fill_value)


def _merge_labels(
    a: None | Ts,
    b: None | Ts,
    op: Callable[[Ts, Ts], Ts],
) -> None | Ts:
    if a is None and a is None:
        return None

    if a is None:
        a = _full_like(b, fill_value=-1)
        return op([a, b])
    if b is None:
        b = _full_like(a, fill_value=-1)
        return op([a, b])

    return op([a, b])


class RetrievalData(ABC, Generic[Ts_co]):
    """Model search results."""

    __slots__ = ("scores", "indices", "labels")
    _expected_dim: int
    _str_sep: str = ""
    _repr_sep: str = ""
    scores: Ts_co
    indices: Ts_co
    labels: None | Ts_co
    meta: dict[str, Any]

    def __init__(
        self,
        scores: Ts_co,
        indices: Ts_co,
        labels: Optional[Ts_co] = None,
        meta: Optional[dict[str, Any]] = None,
        allow_unsafe: bool = False,
    ):
        dim = len(indices.shape)
        # note: only check shapes up to the number of dimensions of the indices. This allows
        # for the scores to have more dimensions than the indices, e.g. for the case of
        # merging two batches.
        if not allow_unsafe and scores.shape[:dim] != indices.shape[:dim]:
            raise ValueError(
                f"The shapes of `scores` and `indices` must match up to the dimension of `indices`, "
                f"but got {_array_repr(scores)} and {_array_repr(indices)}"
            )
        if labels is not None and (scores.shape[:dim] != labels.shape[:dim]):
            raise ValueError("The shapes of `scores` and `labels` must match up to the dimension of `indices`, ")
        if len(scores.shape) != self._expected_dim:
            raise ValueError(
                f"Scores must be {self._expected_dim}D, " f"but got {_array_repr(scores)} and {_array_repr(indices)}"
            )

        self.allow_unsafe = allow_unsafe
        self.scores = scores
        self.indices = indices
        self.labels = labels
        self.meta = meta or {}

    def to(self, target_type: Type[Ts_out]) -> RetrievalData[Ts_out]:
        """Cast a `RetrievalData` object to a different type."""
        output: RetrievalData = type(self)(
            scores=_convert_array(self.scores, target_type),
            indices=_convert_array(self.indices, target_type),
            labels=_convert_array(self.labels, target_type) if self.labels is not None else None,
            meta=copy.copy(self.meta),
        )
        return output

    @abc.abstractmethod
    def __getitem__(self, item: SliceType) -> "RetrievalData":
        """Slice the data."""
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterable["RetrievalData"]:
        """Iterate over the data."""
        ...

    def __len__(self) -> int:
        """Length of the data."""
        return len(self.scores)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data."""
        return self.scores.shape

    def _get_repr_parts(self) -> list[str]:
        return [
            f"{type(self).__name__}[{_type_repr(self.scores)}](",
            f"scores={repr(self.scores)}, ",
            f"indices={repr(self.indices)}, ",
            f"labels={repr(self.labels)}, ",
            f"meta={repr(self.meta)}",
            ")",
        ]

    def __repr__(self) -> str:
        """Representation of the object."""
        parts = self._get_repr_parts()
        return self._repr_sep.join(parts[:-1]) + parts[-1]

    def __str__(self) -> str:
        """String representation of the object."""
        parts = self._get_repr_parts()
        return self._str_sep.join(parts[:-1]) + parts[-1]

    def __eq__(self, other: object) -> bool:
        """Compare two `RetrievalData` objects."""
        if not isinstance(other, type(self)):
            raise NotImplementedError(f"Cannot compare {type(self)} with {type(other)}")
        op = {
            torch.Tensor: torch.all,
            np.ndarray: np.all,
        }[type(self.scores)]
        return op(self.scores == other.scores) and op(self.indices == other.indices)

    def to_dict(self) -> dict[str, None | list[Number]]:
        """Convert to a dictionary."""
        return {
            "scores": self.scores.tolist(),
            "indices": self.indices.tolist(),
            "labels": self.labels.tolist() if self.labels is not None else None,
        }


class RetrievalTuple(RetrievalData[Ts]):
    """A single search result."""

    _expected_dim = 0

    def __getitem__(self, item: Any) -> Any:  # noqa: ANN401
        """Not implemented for single samples."""
        raise NotImplementedError("RetrievalTuple is not iterable")

    def __iter__(self) -> Any:  # noqa: ANN401
        """Not implemented for single samples."""
        raise NotImplementedError("RetrievalTuple is not iterable")


class RetrievalSample(RetrievalData[Ts_co]):
    """A single value of a search result."""

    _expected_dim = 1
    _str_sep: str = ""

    def __getitem__(self, item: int) -> RetrievalTuple:
        """Get a single value from the sample."""
        return RetrievalTuple(
            scores=self.scores[item],
            indices=self.indices[item],
            labels=self.labels[item] if self.labels is not None else None,
        )

    def __iter__(self) -> Iterable[RetrievalTuple[Ts_co]]:
        """Iterate over the sample dimension."""
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other: "RetrievalSample") -> "RetrievalBatch":
        """Concatenate two samples along the sample dimension."""
        return RetrievalBatch(
            scores=_stack_arrays([self.scores, other.scores]),
            indices=_stack_arrays([self.indices, other.indices]),
            labels=_merge_labels(self.labels, other.labels, _stack_arrays),
        )


class RetrievalBatch(RetrievalData[Ts_co]):
    """A batch of search results."""

    _expected_dim = 2
    _str_sep: str = "\n"

    def __getitem__(self, item: int) -> RetrievalSample:
        """Get a single sample from the batch."""
        return RetrievalSample(
            scores=self.scores[item],
            indices=self.indices[item],
            labels=self.labels[item] if self.labels is not None else None,
        )

    def __iter__(self) -> Iterable[RetrievalSample[Ts]]:
        """Iterate over the batch dimension."""
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other: "RetrievalBatch") -> "RetrievalBatch":
        """Concatenate two batches along the batch dimension."""
        return RetrievalBatch(
            scores=_concat_arrays([self.scores, other.scores]),
            indices=_concat_arrays([self.indices, other.indices]),
            labels=_merge_labels(self.labels, other.labels, _concat_arrays),
        )

    def to(self, target_type: Type[Ts_out]) -> RetrievalBatch[Ts_out]:
        """Cast a `RetrievalBatch` object to a different type."""
        return RetrievalBatch(
            scores=_convert_array(self.scores, target_type),
            indices=_convert_array(self.indices, target_type),
            labels=_convert_array(self.labels, target_type) if self.labels is not None else None,
            meta=copy.copy(self.meta),
        )

    def sorted(self) -> RetrievalBatch[Ts_co]:
        """Sort the batch by score in descending order."""
        if isinstance(self.indices, np.ndarray):
            if not isinstance(self.scores, np.ndarray):
                raise TypeError(f"Incomapatible types {type(self.scores)} and {type(self.indices)}")
            sort_ids = np.argsort(self.scores, axis=-1)
            sort_ids = np.flip(sort_ids, axis=-1)
            return RetrievalBatch(
                scores=np.take_along_axis(self.scores, sort_ids, axis=-1),
                indices=np.take_along_axis(self.indices, sort_ids, axis=-1),
                labels=np.take_along_axis(self.labels, sort_ids, axis=-1) if self.labels is not None else None,
                meta=copy.copy(self.meta),
            )
        if isinstance(self.indices, torch.Tensor):
            if not isinstance(self.scores, torch.Tensor):
                raise TypeError(f"Incomapatible types {type(self.scores)} and {type(self.indices)}")
            sort_ids = torch.argsort(self.scores, dim=-1, descending=True)
            return RetrievalBatch(
                scores=torch.gather(self.scores, -1, sort_ids),
                indices=torch.gather(self.indices, -1, sort_ids),
                labels=torch.gather(self.labels, -1, sort_ids) if self.labels is not None else None,
                meta=copy.copy(self.meta),
            )

        raise NotImplementedError(f"Sorting is not implemented for {type(self.scores)}")

    def __mul__(self, value: float) -> RetrievalBatch[Ts_co]:
        """Multiply scores by a value."""
        return RetrievalBatch(
            scores=self.scores * value,
            indices=self.indices,
            labels=self.labels,
            meta=copy.copy(self.meta),
        )


def merge_retrieval_batches(batches: Iterable[RetrievalBatch]) -> RetrievalBatch:
    """Merge a list of `RetrievalBatch` objects into a single `RetrievalBatch` object."""
    batches = list(batches)
    if len(batches) == 0:
        raise ValueError("No batches provided")
    if len(batches) == 1:
        return batches[0]
    if len(batches) > 2:  # noqa: PLR2004
        raise NotImplementedError("Merging more than 2 batches is not implemented")

    first_batch, second_batches = batches

    if first_batch.labels is not None or second_batches.labels is not None:
        raise NotImplementedError("Merging batches with labels is not implemented")

    py_type = type(first_batch.scores)
    new_indices, new_scores = c_tools.merge_search_results(
        a_indices=first_batch.indices,
        a_scores=first_batch.scores,
        b_indices=second_batches.indices,
        b_scores=second_batches.scores,
    )
    return RetrievalBatch(indices=new_indices, scores=new_scores, labels=None).to(py_type)
