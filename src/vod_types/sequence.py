import typing as typ

_T = typ.TypeVar("_T")
_T_co = typ.TypeVar("_T_co", covariant=True)


@typ.runtime_checkable
class Sequence(typ.Protocol[_T_co]):
    """A sequence of data."""

    def __getitem__(self, __it: int) -> _T_co:
        ...

    def __len__(self) -> int:
        ...


@typ.runtime_checkable
class DictsSequence(typ.Protocol[_T]):
    """A sequence of dictionaries."""

    def __getitem__(self, __it: int) -> dict[str, _T]:
        ...

    def __len__(self) -> int:
        ...
