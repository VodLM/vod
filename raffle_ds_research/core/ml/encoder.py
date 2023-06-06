from __future__ import annotations

import importlib
import pathlib
from typing import Literal, Optional

import pydantic
import torch
import transformers
from torch import nn
from typing_extensions import Self, Type

from raffle_ds_research.core.ml import public


def _resolve_import_path(path: str) -> Type:
    modpath, clsname = path.rsplit(".", 1)
    return getattr(importlib.import_module(modpath), clsname)


class TransformerEncoderConfig(pydantic.BaseModel):
    """Configuration for a transformer encoder."""

    model_name: str
    vector_size: Optional[int] = None
    cls_name: str = "AutoModel"
    agg: Literal["mean", "max", "cls"] = "mean"


def _mean_agg(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    return x.sum(dim=-2) / mask.sum(dim=-1, keepdim=True)


def _cls_agg(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    return x[..., 0, :]


def _max_agg(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    return x.max(dim=-2).values


AGGREGATOR_FNS = {
    "mean": _mean_agg,
    "max": _max_agg,
    "cls": _cls_agg,
}


class TransformerEncoder(nn.Module, public.ProtocolEncoder):
    """A transformer encoder."""

    def __init__(
        self,
        model_name: str,
        vector_size: Optional[int] = None,
        cls_name: str = "AutoModel",
        agg: Literal["mean", "max", "cls"] = "mean",
        cache_dir: None | str | pathlib.Path = None,
    ):
        super().__init__()
        self.config = TransformerEncoderConfig(
            model_name=model_name, vector_size=vector_size, cls_name=cls_name, agg=agg
        )
        cls = getattr(transformers, cls_name)
        self.model = cls.from_pretrained(model_name, cache_dir=cache_dir)
        if self.model.config.hidden_size is None:
            raise ValueError("`hidden_size` could not be inferred from the model config.")

        # delete the projection layer to avoid `none` grads
        if hasattr(self.model, "pooler"):
            self.model.pooler = None

        # make the projection layer
        if vector_size is None:
            vector_size = int(self.model.config.hidden_size)

        self.projection = nn.Sequential(
            torch.nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, vector_size),
        )
        self._output_size = vector_size

    def save(self, path: str | pathlib.Path) -> None:
        """Save the encoder."""
        state = {
            "state": self.state_dict(),
            "config": self.config.dict(),
        }
        torch.save(state, path)

    @classmethod
    def load(cls: Type[Self], path: str | pathlib.Path) -> Self:
        """Load the encoder."""
        state = torch.load(path)
        config = TransformerEncoderConfig(**state["config"])
        encoder = cls(**config.dict())
        encoder.load_state_dict(state["state"])
        return encoder

    def forward(self, data: public.TokenizedField) -> torch.Tensor:
        """Embed/encode a tokenized field into a vector."""
        outputs = self.model(data.input_ids, data.attention_mask)
        agg_fn = AGGREGATOR_FNS[self.config.agg]
        hidden_state = agg_fn(outputs.last_hidden_state, data.attention_mask)
        return self.projection(hidden_state)

    def get_output_shape(self, field: Optional[public.TokenizedField] = None) -> tuple[int, ...]:  # noqa: ARG002
        """Get the output shape of the encoder. Set `-1` for unknown dimensions."""
        return (self._output_size,)
