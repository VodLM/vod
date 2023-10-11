import abc
import typing as typ

import pydantic
import torch
import torch.nn
from vod_tools import pretty


class Gradients:
    """Base class for the gradients layer. The gradients layer is a pure function (no torch params)."""

    @abc.abstractmethod
    def __call__(self, intermediate_results: dict) -> dict:
        """Compute the gradients/loss."""
        raise NotImplementedError


class GradientInputs(pydantic.BaseModel):
    """collection of inputs for the supervised gradients model."""

    class Config:
        """pydantic config."""

        arbitrary_types_allowed = True

    hq: None | torch.Tensor = None
    hd: None | torch.Tensor = None
    targets: torch.Tensor = pydantic.Field(
        ...,
        description="Retrieval labels.",
        alias="section.label",
    )
    scores: torch.Tensor = pydantic.Field(
        ...,
        description="Retrieval scores.",
        alias="section.score",
    )
    sparse: None | torch.Tensor = pydantic.Field(
        None,
        description="Sparse retrieval scores.",
        alias="section.sparse",
    )
    dense: None | torch.Tensor = pydantic.Field(
        None,
        description="dense retrieval scores.",
        alias="section.dense",
    )

    def pprint(self, **kws: typ.Any) -> None:
        """Pretty print the inputs."""
        pretty.pprint_batch({k: v for k, v in self.dict().items() if v is not None}, **kws)
