import math
from numbers import Number
from typing import Optional, Union

import torch
from pydantic import BaseModel, Field, PrivateAttr

from .pipe import E, I, O, Pipe


class Sampler(Pipe[I, O, E]):
    total: Optional[int] = Field(
        10,
        description="The total number of examples to sample.",
    )
    index_key: str = Field(
        "section.pid",
        alias="output index key",
    )
    score_key: str = Field(
        "section.score",
        alias="output index key",
    )
    label_key: str = Field(
        "section.label",
        alias="output index key",
    )


class RandomSamplerInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    scores: Union[torch.Tensor, list[list[float]]] = Field(
        ...,
        description="The scores to sample from.",
        alias="section.score",
    )
    pids: Union[torch.Tensor, list[list[int]]] = Field(
        ...,
        description="The indices of the sections",
        alias="section.pid",
    )
    labels: Optional[Union[torch.Tensor, list[list[bool]]]] = Field(
        None,
        description="The labels of the sections",
        alias="section.label",
    )


class RandomSampler(Sampler[RandomSamplerInput, dict, dict]):
    _input_model = PrivateAttr(RandomSamplerInput)

    def _process_batch(
        self, batch: dict, idx: Optional[list[int]] = None, keep_tensors: bool = False, **kwargs
    ) -> dict[str, list[list[Number]]]:
        indices = batch["pids"]
        scores = batch["scores"]
        labels = batch["labels"]
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.bool)
        probs = torch.softmax(scores, dim=-1)
        samples = probs.multinomial(self.total, replacement=False)
        sampled_indices = indices.gather(-1, index=samples)
        sampled_scores = scores.gather(-1, index=samples)
        sampled_labels = labels.gather(-1, index=samples)

        if not keep_tensors:
            sampled_indices = sampled_indices.tolist()
            sampled_scores = sampled_scores.tolist()
            sampled_labels = sampled_labels.to_list()

        return {
            self.score_key: sampled_scores,
            self.index_key: sampled_indices,
            self.label_key: sampled_labels,
        }


if __name__ == "__main__":
    batch = {
        "section.score": [
            [0.1, 0.2, -math.inf, -math.inf],
            [0.1, 0.2, 0.3, 0.4],
        ],
        "section.pid": [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
    }

    sampler = RandomSampler(total=3)
    import rich

    rich.print(sampler(batch))
