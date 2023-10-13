import typing as typ

I = typ.TypeVar("I")  # noqa: E741
O = typ.TypeVar("O")  # noqa: E741


@typ.runtime_checkable
class Pipe(typ.Protocol[I, O]):
    """A pipe is a callable that takes a dict and returns a dict."""

    def __call__(
        self,
        batch: typ.Mapping[str, I],
        idx: None | list[int] = None,
        **kws: typ.Any,
    ) -> dict[str, O]:
        """Apply a transformation to a batch of data."""
        ...


@typ.runtime_checkable
class Collate(typ.Protocol[I, O]):
    """A collate is a callable that takes a list of examples and returns a batch."""

    def __call__(
        self,
        inputs: typ.Iterable[typ.Mapping[str, I]],
        **kws: typ.Any,
    ) -> dict[str, O]:
        """Apply the collate function to a list of examples, transforming it into a batch."""
        ...
