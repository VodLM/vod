from copy import copy
from typing import Protocol, Any, Optional


class Pipe(Protocol):
    """A pipe is a callable that takes a batch and returns a batch."""

    def __call__(self, batch: dict[str, Any], idx: Optional[list[int]] = None, **kwargs: Any) -> dict[str, Any]:
        ...


class Collate(Protocol):
    """A collate is a callable that takes a list of examples and returns a batch."""

    def __call__(self, examples: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        ...


def retain_inputs(pipe: Pipe) -> Pipe:
    """Decorator to retain the inputs to a pipe."""

    def wrapper(batch: dict[str, Any], idx: Optional[list[int]] = None, **kwargs: Any) -> dict[str, Any]:
        input_batch = copy(batch)
        output_batch = pipe(batch, idx, **kwargs)
        input_batch.update(output_batch)
        return input_batch

    return wrapper


def filter_inputs(pipe: Pipe, keys: list[str], strict: bool = True) -> Pipe:
    """Decorator to filter the inputs to a pipe."""

    def wrapper(batch: dict[str, Any], idx: Optional[list[int]] = None, **kwargs: Any) -> dict[str, Any]:
        if strict:
            input_batch = {key: batch[key] for key in keys}
        else:
            input_batch = {key: batch[key] for key in keys if key in batch}
        if len(input_batch) == 0:
            raise KeyError(f"No keys found in batch. Expected keys: `{keys}`. Found keys: `{list(batch.keys())}`")
        output_batch = pipe(input_batch, idx, **kwargs)
        batch.update(output_batch)
        return batch

    return wrapper
