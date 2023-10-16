import abc
import functools
import io
import json
import typing as typ

import numpy as np
import torch
import transformers
import xxhash
from torch import nn
from transformers import modeling_outputs

from .configuration import (
    AggMethod,
    VodBertEncoderConfig,
    VodEncoderConfig,
    VodEncoderInputType,
    VodPoolerConfig,
    VodRobertaEncoderConfig,
    VodT5EncoderConfig,
    VodXLMRobertaEncoderConfig,
)


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


class Aggregator(abc.ABC, nn.Module):
    """Base class for aggregators."""

    def __init__(self) -> None:
        super().__init__()
        self._dtype_marker = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Summarize a sequence of hidden states into a single one."""
        ...


class MeanAgg(Aggregator):
    """Mean aggregator."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG
        sum_mask = mask.sum(dim=-1, keepdim=True)
        x_mean = x.sum(dim=-2) / sum_mask.to(self._dtype_marker.dtype)
        return x_mean.masked_fill(sum_mask <= 0, 0.0)


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


AGGREGATORS: dict[AggMethod, typ.Type[Aggregator]] = {
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
    projection: None | nn.Linear
    activation: None | nn.Module
    norm_fn: None | typ.Callable[[torch.Tensor], torch.Tensor]
    output_vector_size: int
    scaler: nn.Parameter

    def __init__(self, config: dict | VodPoolerConfig, backbone_output_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = VodPoolerConfig(**config)
        self.backbone_output_size = backbone_output_size
        self.aggregator = AGGREGATORS[config.agg_method]()
        if config.projection_size is None:
            self.output_vector_size = self.backbone_output_size
            self.projection = None
        else:
            self.output_vector_size = config.projection_size
            self.projection = nn.Linear(self.backbone_output_size, self.output_vector_size)

        # Activation
        if config.output_activation is None:
            self.activation = None
        else:
            self.activation = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid,
                "gelu": nn.GELU,
            }[config.output_activation]()

        # Normalization
        if config.output_norm is None:
            self.norm_fn = None
        else:
            self.norm_fn = {
                "l2": functools.partial(torch.nn.functional.normalize, p=2),
                "l1": functools.partial(torch.nn.functional.normalize, p=1),
            }[config.output_norm]

        # Temperature
        self.scaler = nn.Parameter(torch.tensor(config.scaler or 1.0), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor, *, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pools the model output and project. Note that the activation is applied last."""
        pooled_output = self.aggregator(hidden_states, attention_mask)
        if self.projection:
            pooled_output = self.projection(pooled_output)
        if self.activation:
            pooled_output = self.activation(pooled_output)
        if self.norm_fn:
            pooled_output = self.norm_fn(pooled_output)
        return pooled_output / self.scaler

    def get_encoding_shape(self) -> tuple[int, ...]:
        """The output dimension of the encoder."""
        if isinstance(self.aggregator, IdentityAgg):
            return (-1, self.output_vector_size)

        return (self.output_vector_size,)


Cfg = typ.TypeVar("Cfg", bound=VodEncoderConfig)


class VodEncoderBase(typ.Generic[Cfg], transformers.PreTrainedModel, abc.ABC):
    """A VOD transformer encoder."""

    config_class: typ.Type[Cfg]
    vod_pooler: VodPooler

    def __init__(self, config: Cfg, *args: typ.Any, **kwargs: typ.Any) -> None:
        super().__init__(config, **kwargs)
        self.vod_pooler = VodPooler(config.pooler, self.config.hidden_size)

    def _backbone_forward(
        self,
        input_ids: None | torch.Tensor = None,
        attention_mask: None | torch.Tensor = None,
        input_type: None | VodEncoderInputType = None,  # noqa: ARG002
        **kwargs: typ.Any,
    ) -> modeling_outputs.BaseModelOutput:
        """Forward the input through the backbone."""
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def forward(
        self,
        input_ids: None | torch.Tensor = None,
        attention_mask: None | torch.Tensor = None,
        return_dict: None | bool = False,
        input_type: None | VodEncoderInputType = None,  # noqa: ARG002
        **kwargs: typ.Any,
    ) -> dict | modeling_outputs.BaseModelOutputWithPooling:
        """Encode the input."""
        output = self._backbone_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_type=input_type,
            **kwargs,
        )
        pooled_output = self.vod_pooler(
            output.last_hidden_state,
            attention_mask=attention_mask,
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

    def get_encoding_shape(self) -> tuple[int, ...]:
        """Get the output shape of the encoder. Set `-1` for unknown dimensions."""
        return self.vod_pooler.get_encoding_shape()

    @property
    def base_name_or_path(self) -> str:
        """The name of the base model."""
        return self.config.name_or_path


class VodBertEncoder(VodEncoderBase[VodBertEncoderConfig], transformers.BertModel):
    """A BERT encoder."""

    config_class = VodBertEncoderConfig


VodBertEncoderConfig.register_for_auto_class()
VodBertEncoder.register_for_auto_class("AutoModel")
transformers.AutoConfig.register("vod_bert_encoder", VodBertEncoderConfig)
transformers.AutoModel.register(VodBertEncoderConfig, VodBertEncoder)


class VodT5Encoder(VodEncoderBase[VodT5EncoderConfig], transformers.T5EncoderModel):
    """A T5 encoder."""

    config_class = VodT5EncoderConfig


VodT5EncoderConfig.register_for_auto_class()
VodT5Encoder.register_for_auto_class("AutoModel")
transformers.AutoConfig.register("vod_t5_encoder", VodT5EncoderConfig)
transformers.AutoModel.register(VodT5EncoderConfig, VodT5Encoder)


class VodRobertaEncoder(VodEncoderBase[VodRobertaEncoderConfig], transformers.RobertaModel):
    """A Roberta encoder."""

    config_class = VodRobertaEncoderConfig


VodRobertaEncoderConfig.register_for_auto_class()
VodRobertaEncoder.register_for_auto_class("AutoModel")
transformers.AutoConfig.register("vod_roberta_encoder", VodRobertaEncoderConfig)
transformers.AutoModel.register(VodRobertaEncoderConfig, VodRobertaEncoder)


class VodXLMRobertaEncoder(VodEncoderBase[VodXLMRobertaEncoderConfig], transformers.RobertaModel):
    """A XLM Roberta encoder."""

    config_class = VodXLMRobertaEncoderConfig


VodXLMRobertaEncoderConfig.register_for_auto_class()
VodXLMRobertaEncoder.register_for_auto_class("AutoModel")
transformers.AutoConfig.register("vod_xlm_roberta_encoder", VodXLMRobertaEncoderConfig)
transformers.AutoModel.register(VodXLMRobertaEncoderConfig, VodXLMRobertaEncoder)


class EmbeddingOnlyOverride(VodEncoderBase):
    """Mixin for embedding-only encoders."""

    def _backbone_forward(
        self: VodEncoderBase,
        input_ids: None | torch.Tensor = None,
        attention_mask: None | torch.Tensor = None,  # noqa: ARG002
        input_type: None | VodEncoderInputType = None,  # noqa: ARG002
        **kwargs: typ.Any,
    ) -> modeling_outputs.BaseModelOutput:
        """Forward the through the embedding layer."""
        try:
            output = self.embeddings(input_ids=input_ids)  # type: ignore
        except AttributeError as exc:
            raise AttributeError(
                f"Expected self (type={type(self).__name__}) to have an `embeddings` attribute."
            ) from exc
        return modeling_outputs.BaseModelOutput(
            last_hidden_state=output,
            hidden_states=None,
            attentions=None,
        )


class VodBertEncoderDebug(EmbeddingOnlyOverride, VodBertEncoder):
    """A BERT encoder for debugging."""

    ...


VodBertEncoderDebug.register_for_auto_class()


class VodT5EncoderDebug(EmbeddingOnlyOverride, VodT5Encoder):
    """A T5 encoder for debugging."""

    ...


VodT5EncoderDebug.register_for_auto_class()


class VodRobertaEncoderDebug(EmbeddingOnlyOverride, VodRobertaEncoder):
    """A Roberta encoder for debugging."""

    ...


VodRobertaEncoderDebug.register_for_auto_class()


class VodXLMRobertaEncoderDebug(EmbeddingOnlyOverride, VodXLMRobertaEncoder):
    """A XLM Roberta encoder for debugging."""

    ...


VodXLMRobertaEncoderDebug.register_for_auto_class()


VodEncoder = VodBertEncoder | VodT5Encoder | VodRobertaEncoder | VodXLMRobertaEncoder
