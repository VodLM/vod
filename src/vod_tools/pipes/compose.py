from __future__ import annotations

from copy import copy
from typing import Any, Optional

from raffle_ds_research.tools.pipes.protocols import Pipe


class Sequential(Pipe):
    """Defines a list of transformations."""

    def __init__(self, list_of_pipes: list[Pipe], with_updates: bool = False, **kwargs: Any):
        if len(list_of_pipes) == 0:
            raise ValueError("The list of pipes must not be empty.")
        if len(kwargs) > 0:
            raise ValueError(f"The Sequential pipe does not accept any kwargs. Found {kwargs}")
        self.pipes = list_of_pipes
        self.with_updates = with_updates

    def __call__(self, batch: dict[str, Any], idx: Optional[list[int]] = None, **kwargs: Any) -> dict[str, Any]:
        """Call a sequence of Pipes."""
        for pipe in self.pipes:
            batch = copy(batch)
            batch_ = pipe(batch, idx, **kwargs)
            if self.with_updates:
                batch.update(batch_)
            else:
                batch = batch_

        return batch
