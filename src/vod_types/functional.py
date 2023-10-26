import typing as typ

I_contra = typ.TypeVar("I_contra", contravariant=True)
O_co = typ.TypeVar("O_co", covariant=True)


@typ.runtime_checkable
class Pipe(typ.Protocol[I_contra, O_co]):
    """A pipe is a callable that takes a dict and returns a dict."""

    def __call__(
        self,
        batch: typ.Mapping[str, I_contra],
        idx: None | list[int] = None,
        **kws: typ.Any,
    ) -> typ.Mapping[str, O_co]:
        """Apply a transformation to a batch of data."""
        ...


@typ.runtime_checkable
class Collate(typ.Protocol[I_contra, O_co]):
    """A collate is a callable that takes a list of examples and returns a batch."""

    def __call__(
        self,
        inputs: typ.Iterable[typ.Mapping[str, I_contra]],
        **kws: typ.Any,
    ) -> typ.Mapping[str, O_co]:
        """Apply the collate function to a list of examples, transforming it into a batch."""
        ...
