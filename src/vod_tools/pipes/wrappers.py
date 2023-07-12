from __future__ import annotations

from copy import copy
from typing import Any, Optional

from src.vod_tools.pipes.protocols import Pipe


def retain_inputs_wrapper(pipe: Pipe) -> Pipe:
    """Retain the inputs along with the pipe outputs."""

    def wrapper(batch: dict[str, Any], idx: Optional[list[int]] = None, **kwargs: Any) -> dict[str, Any]:
        input_batch = copy(batch)
        output_batch = pipe(batch, idx, **kwargs)
        input_batch.update(output_batch)
        return input_batch

    return wrapper


def filter_inputs_wrapper(pipe: Pipe, keys: list[str], strict: bool = True) -> Pipe:
    """Filter the inputs to a pipe."""

    def wrapper(batch: dict[str, Any], idx: Optional[list[int]] = None, **kwargs: Any) -> dict[str, Any]:
        input_batch = {key: batch[key] for key in keys} if strict else {key: batch[key] for key in keys if key in batch}
        if len(input_batch) == 0:
            raise KeyError(f"No keys found in batch. Expected keys: `{keys}`. Found keys: `{list(batch.keys())}`")
        output_batch = pipe(input_batch, idx, **kwargs)
        batch.update(output_batch)
        return batch

    return wrapper


def key_map_wrapper(pipe: Pipe, key_map: dict[str, str]) -> Pipe:
    """Map the keys of a batch to a new set of keys."""

    def wrapper(batch: dict[str, Any], idx: Optional[list[int]] = None, **kwargs: Any) -> dict[str, Any]:
        input_batch = {key_map.get(key, key): batch[key] for key in batch}
        return pipe(input_batch, idx, **kwargs)

    return wrapper


class Partial(Pipe):
    """This class is used to partially apply arguments to a pipe."""

    def __init__(self, pipe: Pipe, **kwargs: Any):
        self.pipe = pipe
        self.kwargs = kwargs

    def __call__(self, batch: dict[str, Any], idx: Optional[list[int]] = None, **kwargs: Any) -> dict[str, Any]:
        """Call the attribute `pipe` with the arguments `kwargs` and `self.kwargs`."""
        return self.pipe(batch, idx=idx, **self.kwargs, **kwargs)
