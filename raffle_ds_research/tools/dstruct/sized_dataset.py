from __future__ import annotations

import typing

import typing_extensions

T = typing.TypeVar("T", covariant=True)
T_co = typing.TypeVar("T_co", covariant=True)

SliceType: typing_extensions.TypeAlias = typing.Union[int, slice, list[int]]


@typing.runtime_checkable
class SizedDataset(typing.Protocol[T_co]):
    """An object that can be indexed."""

    def __getitem__(self, idx: SliceType) -> T_co:
        """Get an item by index."""
        ...

    def __len__(self) -> int:
        """Get the length of the indexable."""
        ...
