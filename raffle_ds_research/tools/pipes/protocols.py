from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class Pipe(Protocol):
    """A pipe is a callable that takes a batch and returns a batch."""

    def __init__(self, *args: Any, **kwargs: Any):
        ...

    def __call__(self, batch: dict[str, Any], idx: Optional[list[int]] = None, **kwargs: Any) -> dict[str, Any]:
        """Apply a transformation to a batch of data."""
        ...


@runtime_checkable
class Collate(Protocol):
    """A collate is a callable that takes a list of examples and returns a batch."""

    def __init__(self, *args: Any, **kwargs: Any):
        ...

    def __call__(self, examples: Iterable[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Apply the collate function to a list of examples, transforming it into a batch."""
        ...
