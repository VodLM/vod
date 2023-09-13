import typing as typ
from collections import abc

import datasets

T = typ.TypeVar("T")
T_co = typ.TypeVar("T_co", covariant=True)

SliceType: typ.TypeAlias = typ.Union[int, slice, list[int]]


class Sequence(abc.Sequence[T_co]):
    """Similar to `collections.abc.Sequence`."""

    def __getitem__(self, idx: SliceType) -> T_co:
        """Get an item by index."""
        ...

    def __len__(self) -> int:
        """Get the length of the indexable."""
        ...


class DictsSequence(Sequence[dict[str, T]]):
    """A sequence of dictionaries."""


# Register `datasets.Dataset` as a `SequenceDict`.
DictsSequence.register(datasets.Dataset)  # type: ignore
DictsSequence.register(DictsSequence)  # type: ignore
