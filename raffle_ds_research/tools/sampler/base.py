import abc
from dataclasses import dataclass
from typing import Any, Optional

import pydantic
import torch


@dataclass
class Samples(pydantic.BaseModel):
    """A class to store the samples"""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        allow_mutation = False

    indices: torch.Tensor = pydantic.Field(..., description="Indices of the samples.")
    log_weights: torch.Tensor = pydantic.Field(None, description="Weight for each sample (importance sampling).")

    @pydantic.root_validator(pre=True)
    def _set_empty_weights(self, values: dict[str, Any]) -> dict[str, torch.Tensor]:
        indices = values["indices"]
        if not isinstance(indices, torch.Tensor):
            raise TypeError("`indices` must be a Tensor.")
        log_weights = values.get("log_weights", None)
        if log_weights is None:
            log_weights = torch.zeros_like(indices, dtype=torch.float32, device=indices.device)

        return {
            "indices": indices,
            "log_weights": log_weights,
        }


class Sampler(abc.ABC):
    """A Categorical distribution."""

    @abc.abstractmethod
    def __call__(
        self,
        scores: torch.Tensor,
        *,
        label: Optional[torch.Tensor] = None,
        n: int = 3,
    ) -> Samples:
        """Sample from the distribution defined `p(z) = Softmax(scores)`"""
        raise NotImplementedError


class Categorical(Sampler):
    def __init__(self, replace: bool = True):
        self.replace = replace
        if not replace:
            raise NotImplementedError("Priority simpling is not implemented for replacement=False.")

    def __call__(
        self,
        scores: torch.Tensor,
        *,
        label: Optional[torch.Tensor] = None,
        n: int = 3,
    ) -> Samples:
        shape = scores.shape
        scores = scores.view(-1, shape[-1])
        z = scores.multinomial(n, replacement=self.replace)
        return Samples(indices=z)
