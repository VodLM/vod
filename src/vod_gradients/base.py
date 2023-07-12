from __future__ import annotations

import abc
from typing import Any, Callable, Optional

import lightning as L
import pydantic
import torch
import torch.nn
from vod_tools import pipes


class Gradients:
    """Base class for the gradients layer. The gradients layer is a pure function (no torch params)."""

    @abc.abstractmethod
    def __call__(self, intermediate_results: dict) -> dict:
        """Compute the gradients/loss."""
        raise NotImplementedError

    def forward_backward(
        self,
        batch: dict[str, torch.Tensor],
        fwd_fn: None | Callable[[dict], dict],
        fabric: None | L.Fabric = None,
        loss_scaler: Optional[float] = None,
        backward_kwargs: Optional[dict] = None,
        no_backward_sync: bool = False,
        fwd_kwargs: None | dict = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass with a backward pass."""
        fwd_kwargs = fwd_kwargs or {}
        grad_output = fwd_fn(batch, **fwd_kwargs) if fwd_fn is not None else {}

        # compute the loss
        loss = grad_output["loss"]
        if loss_scaler is not None:
            loss *= loss_scaler

        # backward pass
        backward_kwargs = backward_kwargs or {}
        if fabric is None:
            loss.backward(**backward_kwargs)
        else:
            with fabric.no_backward_sync(fwd_fn, enabled=no_backward_sync):  # type: ignore
                fabric.backward(loss, **backward_kwargs)

        return grad_output


class GradientInputs(pydantic.BaseModel):
    """collection of inputs for the supervised gradients model."""

    class Config:
        """pydantic config."""

        arbitrary_types_allowed = True

    hq: Optional[torch.Tensor] = None
    hd: Optional[torch.Tensor] = None
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
    bm25: Optional[torch.Tensor] = pydantic.Field(
        None,
        description="bm25 Retrieval scores.",
        alias="section.bm25",
    )
    faiss: Optional[torch.Tensor] = pydantic.Field(
        None,
        description="faiss Retrieval scores.",
        alias="section.faiss",
    )

    # Precomputed logprobs from the model.
    pre_logits: Optional[torch.Tensor] = pydantic.Field(
        None,
        description="Precomputed logprobs from the model.",
        alias="section.pre_logits",
    )

    pre_n_positive: Optional[torch.Tensor] = pydantic.Field(
        None,
        description="Precomputed total number of positive documents.",
        alias="section.pre_n_positive",
    )

    def pprint(self, **kwargs: Any) -> None:
        """Pretty print the inputs."""
        pipes.pprint_batch({k: v for k, v in self.dict().items() if v is not None}, **kwargs)
