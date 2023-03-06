import json
from typing import Optional

from datasets.fingerprint import Hasher, hashregister
from pydantic import Field

from .pipe import Pipe


class Sequential(Pipe):
    pipes: list[Pipe] = Field(..., description="The pipes to run sequentially.")

    def _process_batch(self, batch: dict, idx: Optional[list[int]] = None, **kwargs) -> dict:
        for pipe in self.pipes:
            batch = pipe(batch, idx=idx, **kwargs)
        return batch


@hashregister(Sequential)
def _hash_sequential(hasher: Hasher, value: Sequential):
    data = value.dict()
    data["pipes"] = {i: hasher.hash(p) for i, p in enumerate(data["pipes"])}
    h = hasher.hash_bytes(json.dumps(data, sort_keys=True).encode("utf-8"))
    return h
