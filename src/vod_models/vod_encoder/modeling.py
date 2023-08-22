from __future__ import annotations

import functools
import io
import json
from typing import Any, Callable, Generic, Optional, TypeVar

import numpy as np
import torch
import transformers
import xxhash
from torch import nn
from transformers import modeling_outputs
from typing_extensions import Self, Type

from .configuration import AggMethod, VodEncoderConfig, VodEncoderInputType, VodPoolerConfig


def _serialize_tensor(x: torch.Tensor | np.ndarray) -> bytes:
    """Convert a torch.Tensor into a bytes object."""
    buff = io.BytesIO()
    if isinstance(x, torch.Tensor):
        if x.is_sparse:
            x = x.to_dense()
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float16)
        x = x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

    np.savez(buff, x)
    buff.seek(0)
    return buff.read()


def _compute_fingerprint(model: transformers.PreTrainedModel) -> str:
    """Get a fingerprint of the encoder."""
    hasher = xxhash.xxh64()

    # Config
    hasher.update(json.dumps(model.config.to_dict()).encode())

    # Tensors
    state = model.state_dict()
    hasher.update(type(model).__name__)
    for k, v in sorted(state.items(), key=lambda x: x[0]):
        hasher.update(k)
        u = _serialize_tensor(v)
        hasher.update(u)

    return hasher.hexdigest()


class Aggregator(nn.Module):
    """Base class for aggregators."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Summarize a sequence of hidden states into a single one."""
        raise NotImplementedError()


class MeanAgg(Aggregator):
    """Mean aggregator."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG
        sum_mask = mask.sum(dim=-1, keepdim=True)
        x_mean = x.sum(dim=-2) / sum_mask.float()
        return torch.where(sum_mask > 0, x_mean, 0.0)


class ClsAgg(Aggregator):
    """Returns the vector at the CLS token."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG
        return x[..., 0, :]


class MaxAgg(Aggregator):
    """Returns the vector with the largest norm."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG
        vec_norms = torch.norm(x, dim=-1, keepdim=True)
        return x.gather(dim=-2, index=vec_norms.argmax(dim=-2, keepdim=True))


class IdentityAgg(Aggregator):
    """Identity aggregator (no aggregation)."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG
        return x


AGGREGATORS: dict[AggMethod, Type[Aggregator]] = {
    "mean": MeanAgg,
    "max": MaxAgg,
    "cls": ClsAgg,
    "none": IdentityAgg,
}


class VodPooler(torch.nn.Module):
    """Pool the hidden states of a transformer encoder.

    See `https://github.com/huggingface/transformers/blob/91d7df58b6537d385e90578dac40204cb550f706/src/transformers/models/bert/modeling_bert.py#L654C7-L654C17`
    """

    aggregator: Aggregator
    projection: Optional[nn.Linear]
    activation: Optional[nn.Module]
    norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
    output_vector_size: int

    def __init__(self, config: VodPoolerConfig, backbone_output_size: int):
        super().__init__()
        self.backbone_output_size = backbone_output_size
        self.aggregator = AGGREGATORS[config.agg_method]()
        if config.projection_size is None:
            self.output_vector_size = self.backbone_output_size
            self.projection = None
        else:
            self.output_vector_size = config.projection_size
            self.projection = nn.Linear(self.backbone_output_size, self.output_vector_size)

        # Activation
        self.activation = (
            {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid,
                "gelu": nn.GELU,
            }[config.output_activation]()
            if config.output_activation
            else None
        )

        # Normalization
        self.norm_fn = (
            {
                "l2": functools.partial(torch.nn.functional.normalize, p=2),
                "l1": functools.partial(torch.nn.functional.normalize, p=1),
            }[config.output_norm]
            if config.output_norm
            else None
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pools the model output and project. Note that the activation is applied last."""
        pooled_output = self.aggregator(hidden_states, attention_mask)
        if self.projection:
            pooled_output = self.projection(pooled_output)
        if self.activation:
            pooled_output = self.activation(pooled_output)
        if self.norm_fn:
            pooled_output = self.norm_fn(pooled_output)
        return pooled_output

    def get_output_shape(self) -> tuple[int, ...]:
        """The output dimension of the encoder."""
        if isinstance(self.aggregator, IdentityAgg):
            return (-1, self.output_vector_size)

        return (self.output_vector_size,)


Bck = TypeVar("Bck", bound=transformers.PreTrainedModel)


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


class VodEncoder(Generic[Bck], transformers.PreTrainedModel, torch.nn.Module):
    """A transformer encoder."""

    config_class = VodEncoderConfig
    backbone: Bck
    pooler: VodPooler

    def __init__(
        self,
        config: VodEncoderConfig,
        backbone: Optional[Bck] = None,
    ):
        super().__init__(config)
        self.backbone = backbone or transformers.AutoModel.from_config(config.backbone)  # type: ignore
        self.pooler = VodPooler(config.pooler, self.backbone.config.hidden_size)

    def _backbone_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> modeling_outputs.BaseModelOutput:
        """Forward the input through the backbone."""
        return self.backbone.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        input_type: Optional[VodEncoderInputType] = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> dict | modeling_outputs.BaseModelOutputWithPooling:
        """Encode the input."""
        output = self._backbone_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_type=input_type,
            **kwargs,
        )
        pooled_output = self.pooler(
            output.last_hidden_state,
            attention_mask,
        )
        if return_dict:
            return {
                "last_hidden_state": output.last_hidden_state,
                "hidden_states": output.hidden_states,
                "pooler_output": pooled_output,
                "attentions": output.attentions,
            }

        return modeling_outputs.BaseModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            pooler_output=pooled_output,
            attentions=output.attentions,
        )

    def get_fingerprint(self) -> str:
        """Get a fingerprint of the encoder."""
        return _compute_fingerprint(self)

    def get_output_shape(self) -> tuple[int, ...]:
        """Get the output shape of the encoder. Set `-1` for unknown dimensions."""
        return self.pooler.get_output_shape()

    @property
    def base_name_or_path(self) -> str:
        """The name of the base model."""
        return self.backbone.config.name_or_path

    @classmethod
    def from_pretrained_backbone(
        cls: Type[Self],
        name_or_path: str,
        pooler_config: Optional[dict | VodPoolerConfig] = None,
        backbone_cls: Optional[str | Type[transformers.PreTrainedModel]] = None,
        **kwargs: Any,
    ) -> Self:
        """Instantiate a new model from a pretrained backbone."""
        _b_cls = backbone_cls or transformers.AutoModel
        if isinstance(_b_cls, str):
            _b_cls = getattr(transformers, _b_cls)
        kwargs = _translate_config(name_or_path, kwargs)
        backbone: transformers.PreTrainedModel = _b_cls.from_pretrained(name_or_path, **kwargs)  # type: ignore
        pooler_config = pooler_config or VodPoolerConfig()
        config = cls.config_class(backbone=backbone.config, pooler=pooler_config)  # type: ignore
        return cls(config=config, backbone=backbone)


VodEncoder.register_for_auto_class()
VodEncoder.register_for_auto_class("AutoModel")


class VodDebugEncoder(VodEncoder):
    """A lightweight encoder that can be used for debugging."""

    def _backbone_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> modeling_outputs.BaseModelOutput:
        """Forward the through the embedding layer."""
        output = self.backbone.embeddings(input_ids=input_ids)
        return modeling_outputs.BaseModelOutput(
            last_hidden_state=output,
            hidden_states=None,
            attentions=None,
        )


VodDebugEncoder.register_for_auto_class()
VodDebugEncoder.register_for_auto_class("AutoModel")
