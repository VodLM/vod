from __future__ import annotations

import pathlib
from typing import Literal, Optional

import pydantic
import torch
import transformers
from torch import nn
from typing_extensions import Self, Type, TypeAlias
from vod_tools import interfaces

AggMethod: TypeAlias = Literal["mean", "max", "cls"]


class TransformerEncoderConfig(pydantic.BaseModel):
    """Configuration for a transformer encoder."""

    model_name: str
    vector_size: Optional[int] = None
    cls_name: str = "AutoModel"
    agg: AggMethod = "mean"
    model_config: Optional[dict] = None


# @torch.jit.script  # type: ignore
def _mean_agg(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    sum_mask = mask.sum(dim=-1, keepdim=True)
    x_mean = x.sum(dim=-2) / sum_mask.float()
    return torch.where(sum_mask > 0, x_mean, 0.0)


@torch.jit.script  # type: ignore
def _cls_agg(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    return x[..., 0, :]


@torch.jit.script  # type: ignore
def _max_agg(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    return x.max(dim=-2).values


AGGREGATOR_FNS = {
    "mean": _mean_agg,
    "max": _max_agg,
    "cls": _cls_agg,
}


class EncoderPooler(nn.Module):
    """Pool the hidden states of a transformer encoder.

    See `https://github.com/huggingface/transformers/blob/91d7df58b6537d385e90578dac40204cb550f706/src/transformers/models/bert/modeling_bert.py#L654C7-L654C17`
    """

    output_size: int

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        agg_method: AggMethod = "mean",
        std_init: Optional[float] = None,
    ):
        super().__init__()
        self.output_size = output_size
        self.dense = nn.Linear(hidden_size, output_size)
        if std_init is not None:
            nn.init.normal_(self.dense.weight, std=std_init)

        self.activation = nn.Tanh()
        self.agg_fn = AGGREGATOR_FNS[agg_method]

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pools the model output and project. Note that the activation is applied last."""
        hidden_states = self.agg_fn(hidden_states, attention_mask)
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def _translate_config(model_name: str, config: None | dict) -> dict:
    """Translate the config to a format that transformers can understand."""
    if config is None:
        return {}

    config = config.copy()
    if "bert" in model_name and "dropout" in config:
        dropout_prob = config.pop("dropout")
        config["hidden_dropout_prob"] = dropout_prob
        config["attention_probs_dropout_prob"] = dropout_prob
        return config

    if "t5" in model_name and "dropout" in config:
        dropout_prob = config.pop("dropout")
        config["dropout_rate"] = dropout_prob

    return config


class TransformerEncoder(nn.Module, interfaces.ProtocolEncoder):
    """A transformer encoder."""

    def __init__(
        self,
        model_name: str,
        model_config: Optional[dict] = None,
        vector_size: Optional[int] = None,
        cls_name: str = "AutoModel",
        agg: Literal["mean", "max", "cls"] = "mean",
        cache_dir: None | str | pathlib.Path = None,
        std_init: Optional[float] = None,
    ):
        super().__init__()
        self.config = TransformerEncoderConfig(
            model_name=model_name,
            model_config=model_config,
            vector_size=vector_size,
            cls_name=cls_name,
            agg=agg,
        )
        cls = getattr(transformers, cls_name)
        cfg = _translate_config(model_name, model_config)
        self.backbone = cls.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **cfg,
        )
        if self.backbone.config.hidden_size is None:
            raise ValueError("`hidden_size` could not be inferred from the model config.")

        # Delete the projection layer to avoid `none` grads
        if hasattr(self.backbone, "pooler"):
            self.backbone.pooler = None

        # Define the output dim
        if vector_size is None:
            vector_size = int(self.backbone.config.hidden_size)

        # Projection layer
        self.pooler = EncoderPooler(self.backbone.config.hidden_size, vector_size, agg, std_init=std_init)

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        field: Optional[interfaces.FieldType] = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Embed/encode a tokenized field into a vector."""
        outputs = self.backbone(input_ids, attention_mask)
        return self.pooler(outputs.last_hidden_state, attention_mask)

    def get_output_shape(self, field: Optional[interfaces.TokenizedField] = None) -> tuple[int, ...]:  # noqa: ARG002
        """Get the output shape of the encoder. Set `-1` for unknown dimensions."""
        return (self.pooler.output_size,)
