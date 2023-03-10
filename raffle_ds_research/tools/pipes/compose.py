from __future__ import annotations

from copy import copy
from functools import partial
from typing import Any, Optional

from raffle_ds_research.tools.pipes.protocols import Pipe


class Sequential(object):
    def __init__(self, pipes: list[Pipe | partial], with_updates: bool = False):
        self.pipes = pipes
        self.with_updates = with_updates

    def __call__(self, batch: dict[str, Any], idx: Optional[list[int]] = None, **kwargs: Any) -> dict[str, Any]:
        for pipe in self.pipes:
            batch = copy(batch)
            batch_ = pipe(batch, idx, **kwargs)
            if self.with_updates:
                batch.update(batch_)
            else:
                batch = batch_

        return batch
