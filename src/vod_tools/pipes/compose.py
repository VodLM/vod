import typing as typ
from copy import copy

from vod_types.functional import Pipe


class Sequential(Pipe):
    """Defines a list of transformations."""

    def __init__(self, list_of_pipes: list[Pipe], with_updates: bool = False, **kws: typ.Any):
        if len(list_of_pipes) == 0:
            raise ValueError("The list of pipes must not be empty.")
        if len(kws) > 0:
            raise ValueError(f"The Sequential pipe does not accept any kwargs. Found {kws}")
        self.pipes = list_of_pipes
        self.with_updates = with_updates

    def __call__(self, batch: dict[str, typ.Any], idx: None | list[int] = None, **kws: typ.Any) -> dict[str, typ.Any]:
        """Call a sequence of Pipes."""
        for pipe in self.pipes:
            batch = copy(batch)
            batch_ = pipe(batch, idx, **kws)
            if self.with_updates:
                batch.update(batch_)
            else:
                batch = batch_

        return batch
