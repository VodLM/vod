import abc
import typing as typ

import lightning as L
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

    def forward_backward(
        self,
        batch: dict[str, torch.Tensor],
        fwd_fn: None | typ.Callable[[dict], dict],
        fabric: None | L.Fabric = None,
        loss_scaler: None | float = None,
        backward_kws: None | dict[str, typ.Any] = None,
        no_backward_sync: bool = False,
        fwd_kws: None | dict = None,
        **kws: typ.Any,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass with a backward pass."""
        fwd_kws = fwd_kws or {}
        grad_output = fwd_fn(batch, **fwd_kws) if fwd_fn is not None else {}

        # compute the loss
        loss = grad_output["loss"]
        if loss_scaler is not None:
            loss *= loss_scaler

        # backward pass
        backward_kws = backward_kws or {}
        if fabric is None:
            loss.backward(**backward_kws)
        else:
            with fabric.no_backward_sync(fwd_fn, enabled=no_backward_sync):  # type: ignore
                fabric.backward(loss, **backward_kws)

        return grad_output


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
