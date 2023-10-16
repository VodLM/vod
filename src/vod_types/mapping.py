import typing as typ
from collections import abc

T = typ.TypeVar("T")


class MappingMixin(abc.Mapping[str, T]):
    """A Structured dictionary structure."""

    def __getitem__(self, item: str) -> T:
        return self.__dict__[item]

    def __iter__(self) -> typ.Generator[str, None, None]:
        """Iterate through keys. Required to implement for abc.Mapping."""
        yield from self.__dict__.keys()

    def __len__(self) -> int:
        """Required to implement for abc.Mapping."""
        return len(self.__dict__)

    def values(self) -> typ.Iterable[T]:
        """Return the values of dictionary."""
        return self.__dict__.values()

    def keys(self) -> typ.Iterable[str]:
        """Return the keys of dictionary."""
        return self.__dict__.keys()

    def items(self) -> typ.Iterable[tuple[str, T]]:
        """Return the items of dictionary."""
        yield from self.__dict__.items()

    def dict(self) -> dict[str, T]:
        """Return a dict representation."""
        return self.__dict__.copy()
