import typing as typ

import datasets
import numpy as np
import torch
from vod_types.sequence import Sequence, SliceType, T_co

T = typ.TypeVar("T")


class ConcatenatedSequences(Sequence[T_co]):
    """A concatenation of sequences."""

    parts: list[Sequence[T_co]]

    def __init__(self, parts: typ.Iterable[Sequence[T_co]]) -> None:
        self.parts = list(parts)

    def __getitem__(self, idx: SliceType) -> T_co:
        """Get an item by index."""
        if isinstance(idx, (int, np.int64, np.int32, np.uint64, np.uint32)):  # type: ignore
            return _get_item_int(idx, self.parts)

        if isinstance(idx, slice):
            if idx.start is None:
                idx = slice(0, idx.stop, idx.step)
            if idx.stop is None:
                idx = slice(idx.start, len(self), idx.step)
            chunks = list(_get_interset_slice(idx, self.parts))
            chunks = (p[i] for i, p in chunks)
            return _join_results(chunks)

        return _stack_results([self.__getitem__(i) for i in idx])

    def __len__(self) -> int:
        """Get the length of the indexable."""
        return sum(len(part) for part in self.parts)

    def __str__(self) -> str:
        """Get a string representation of the indexable."""
        return f"{type(self).__name__}(parts={self.parts})"

    def __repr__(self) -> str:
        """Get a string representation of the indexable."""
        return str(self)


def _get_item_int(idx: int, parts: typ.Iterable[Sequence[T]]) -> T:
    i, part = _get_intersect(idx, parts)
    return part[i]


def _get_intersect(idx: int, parts: typ.Iterable[Sequence[T]]) -> tuple[int, Sequence[T]]:
    if idx < 0:
        raise NotImplementedError("Negative indices are not supported.")
    for part in parts:
        if idx < len(part):
            return idx, part
        idx -= len(part)

    raise IndexError(f"Index {idx} out of range.")


def _get_interset_slice(
    idx: slice,
    parts: typ.Iterable[Sequence[T]],
) -> typ.Iterable[tuple[slice, Sequence[T]]]:
    """Return the list of intersecting slices with their relative slices."""
    total_length = sum([len(p) for p in parts])
    if idx.step is not None and idx.step != 1:
        raise NotImplementedError("Slices with steps are not supported.")
    if idx.start == idx.stop:
        raise NotImplementedError("Slices with start == stop are not supported.")
    if idx.start > idx.stop:
        raise NotImplementedError("Slices with start > stop are not supported.")
    if idx.start < 0 or idx.stop < 0:
        raise NotImplementedError("Negative indices are not supported.")
    if idx.start is None:
        idx = slice(0, idx.stop, idx.step)
    if idx.stop is None:
        idx = slice(idx.start, total_length, idx.step)

    offset = 0
    for part in parts:
        start = idx.start - offset
        stop = idx.stop - offset
        if start < len(part) and stop > 0:
            yield slice(max(0, start), min(len(part), stop), idx.step), part

        offset += len(part)


def _join_results(results: typ.Iterable[T]) -> T:
    results = list(results)
    if isinstance(results[0], list):
        return [el for result in results for el in result]  # type: ignore
    if isinstance(results[0], dict):
        keys = results[0].keys()
        return {key: _join_results([result[key] for result in results]) for key in keys}  # type: ignore
    if isinstance(results[0], np.ndarray):
        return np.concatenate(results)  # type: ignore
    if isinstance(results[0], torch.Tensor):
        return torch.cat(results)  # type: ignore

    raise TypeError(f"Unsupported type {type(results[0])}")


def _stack_results(results: typ.Iterable[T]) -> T:
    results = list(results)
    if isinstance(results[0], list):
        return list(results)  # type: ignore
    if isinstance(results[0], dict):
        keys = results[0].keys()
        return {key: _stack_results([result[key] for result in results]) for key in keys}  # type: ignore
    if isinstance(results[0], np.ndarray):
        return np.stack(results)  # type: ignore
    if isinstance(results[0], torch.Tensor):
        return torch.stack(results)  # type: ignore

    return results  # type: ignore


@datasets.fingerprint.hashregister(ConcatenatedSequences)
def _hash_partitioned_indexable(
    hasher: datasets.fingerprint.Hasher,  # noqa: ARG001
    value: ConcatenatedSequences,  # noqa: ARG001
) -> str:  # noqa: ARG001
    hasher_ = datasets.fingerprint.Hasher()
    for part in value.parts:
        hasher_.update(hasher_.hash(part))
    return hasher_.hexdigest()
