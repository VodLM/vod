from __future__ import annotations

import functools
import math
import pathlib
import warnings
from typing import Any, Literal, Optional

import pydantic
import torch
import transformers
from datasets import fingerprint
from torch import nn
from typing_extensions import Self, Type, TypeAlias
from vod_tools import interfaces
from vod_tools.misc.tensor_tools import serialize_tensor

AggMethod: TypeAlias = Literal["mean", "max", "cls", "attention"]


class TransformerEncoderConfig(pydantic.BaseModel):
    """Configuration for a transformer encoder."""

    class Config:
        """Pydantic config."""

        extra = "forbid"
        allow_mutation = False

    model_name: str
    vector_size: Optional[int] = None
    vector_norm: Optional[str] = None
    activation: Optional[str] = None
    cls_name: str = "AutoModel"
    agg: AggMethod = "mean"
    model_config: Optional[dict] = None


class Aggregator(nn.Module):
    """Base class for aggregators."""

    def __init__(self, config: transformers.PretrainedConfig) -> None:  # noqa: ARG002
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Summarize a sequence of hidden states into a single one."""
        raise NotImplementedError()


class _MeanAgg(Aggregator):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG
        sum_mask = mask.sum(dim=-1, keepdim=True)
        x_mean = x.sum(dim=-2) / sum_mask.float()
        return torch.where(sum_mask > 0, x_mean, 0.0)


class _ClsAgg(Aggregator):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG
        return x[..., 0, :]


class _MaxAgg(Aggregator):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG
        return x.max(dim=-2).values


class _AttentionAgg(Aggregator):
    """Attention-based aggregation."""

    def __init__(self, config: transformers.PretrainedConfig):
        super().__init__(config)

        # Parse config
        act_type = _fetch_agg(config)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads

        if hidden_size % num_heads != 0:
            # get the lowest nearest `num_heads` value that divides `hidden_size`
            num_heads = math.gcd(hidden_size, math.ceil(hidden_size / num_heads) * num_heads)
            warnings.warn(
                f"num_heads was changed from `{config.num_attention_heads}` to `{num_heads}` "
                f"to divide hidden_size `{hidden_size}`.",
                stacklevel=2,
            )

        # Set the module params
        self.num_heads = num_heads
        self.query_values = nn.Linear(hidden_size, hidden_size + num_heads)
        self.activation = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
            "gated-gelu": nn.GELU,
        }[act_type]()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Summarize a sequence of hidden states into a single one."""
        qv = self.query_values(x)
        queries, values = qv[..., : self.num_heads], qv[..., self.num_heads :]
        queries = torch.where(mask.unsqueeze(-1) > 0, queries, -math.inf).softmax(dim=-2)
        values = values.view(*values.shape[:-1], self.num_heads, -1)
        y = torch.einsum("...th,...thd->...hd", queries, values)
        y = y.view(*y.shape[:-2], -1)
        y = self.activation(y)
        return y


def _fetch_agg(config: transformers.PretrainedConfig) -> str:
    """Fetch the aggregation method."""
    if isinstance(config, transformers.T5Config):
        return config.feed_forward_proj

    return config.hidden_act


AGGREGATOR_FNS = {
    "mean": _MeanAgg,
    "max": _MaxAgg,
    "cls": _ClsAgg,
    "attention": _AttentionAgg,
}


class EncoderPooler(nn.Module):
    """Pool the hidden states of a transformer encoder.

    See `https://github.com/huggingface/transformers/blob/91d7df58b6537d385e90578dac40204cb550f706/src/transformers/models/bert/modeling_bert.py#L654C7-L654C17`
    """

    output_size: int

    def __init__(
        self,
        config: transformers.PretrainedConfig,
        output_size: None | int,
        agg_method: AggMethod = "mean",
        vector_norm: None | str = None,
        activation: None | str = None,
    ):
        super().__init__()
        self.agg_fn = AGGREGATOR_FNS[agg_method](config)
        if output_size is None:
            self.output_size = config.hidden_size
            self.dense = None
        else:
            self.output_size = output_size
            self.dense = nn.Linear(config.hidden_size, self.output_size)

        # Activation
        self.activation = (
            {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid,
                "gelu": nn.GELU,
            }[activation]()
            if activation
            else None
        )

        # Normalization
        if vector_norm is None:
            self.vector_norm = None
        elif vector_norm == "l2":
            self.vector_norm = functools.partial(torch.nn.functional.normalize, p=2)
        else:
            raise ValueError(f"Unknown vector norm `{vector_norm}`.")

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pools the model output and project. Note that the activation is applied last."""
        pooled_output = self.agg_fn(hidden_states, attention_mask)
        if self.dense:
            pooled_output = self.dense(pooled_output)
        if self.activation:
            pooled_output = self.activation(pooled_output)
        if self.vector_norm:
            pooled_output = self.vector_norm(pooled_output)
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

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        model_config: Optional[dict] = None,
        vector_size: Optional[int] = None,
        vector_norm: Optional[str] = None,
        activation: Optional[str] = None,
        cls_name: str = "AutoModel",
        agg: Literal["mean", "max", "cls"] = "mean",
        cache_dir: None | str | pathlib.Path = None,
    ):
        super().__init__()
        self.config = TransformerEncoderConfig(
            model_name=model_name,
            model_config=model_config,
            vector_size=vector_size,
            vector_norm=vector_norm,
            activation=activation,
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

        # Projection layer
        self.pooler = EncoderPooler(
            self.backbone.config,
            output_size=vector_size,
            agg_method=agg,
            vector_norm=vector_norm,
            activation=activation,
        )

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

    def get_fingerprint(self) -> str:
        """Get a fingerprint of the encoder."""
        hasher = fingerprint.Hasher()

        # Config
        hasher.update(self.config.json().encode())

        # Tensors
        state = self.state_dict()
        hasher.update(type(self).__name__)
        for k, v in sorted(state.items(), key=lambda x: x[0]):
            hasher.update(k)
            u = serialize_tensor(v)
            hasher.update(u)

        return hasher.hexdigest()


class TransformerEncoderDebug(TransformerEncoder):
    """A transformer encoder restricted to the embedding layer only."""

    def __init__(self, *args: Any, **kwds: Any):
        super().__init__(*args, **kwds)
        self.backbone = self.backbone.embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        field: Optional[interfaces.FieldType] = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Embed/encode a tokenized field into a vector."""
        embeddings = self.backbone(input_ids)
        return self.pooler(embeddings, attention_mask)
