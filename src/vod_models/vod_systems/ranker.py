import copy
import functools
import math
import typing as typ

import omegaconf as omg
import torch
import vod_configs
import vod_gradients
from datasets.fingerprint import Hasher, hashregister
from transformers import modeling_outputs
from vod_models import vod_encoder  # type: ignore
from vod_models.monitor import RetrievalMonitor
from vod_models.support import FIELD_MAPPING, apply_tweaks
from vod_tools import fingerprint

from .base import VodSystem


class Ranker(VodSystem):
    """Deep ranking model using a Transformer encoder as a backbone."""

    _output_size: int
    encoder: vod_encoder.VodEncoder

    def __init__(
        self,
        encoder: vod_encoder.VodEncoder,
        gradients: vod_gradients.Gradients,
        optimizer: None | dict | omg.DictConfig | functools.partial = None,
        scheduler: None | dict | omg.DictConfig | functools.partial = None,
        monitor: None | RetrievalMonitor = None,
        tweaks: None | dict | omg.DictConfig | vod_configs.TweaksConfig = None,
    ):
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        self.gradients = gradients
        self.monitor = monitor

        # Prepare the encoder with optional optimizations
        if not isinstance(tweaks, vod_configs.TweaksConfig):
            tweaks = vod_configs.TweaksConfig.parse(**tweaks)  # type: ignore

        self.encoder = apply_tweaks(encoder, tweaks)  # type: ignore

    def get_encoding_shape(self) -> tuple[int, ...]:
        """Dimension of the model output."""
        return self.encoder.get_encoding_shape()

    def get_fingerprint(self) -> str:
        """Return a fingerprint of the model."""
        try:
            return self.encoder.get_fingerprint()
        except AttributeError:
            return fingerprint.fingerprint_torch_module(self)

    def _forward_encoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        original_shape = input_ids.shape
        input_ids = input_ids.view(-1, original_shape[-1])
        attention_mask = attention_mask.view(-1, original_shape[-1])
        output: modeling_outputs.BaseModelOutputWithPooling = self.encoder(input_ids, attention_mask)
        embedding = output.pooler_output
        embedding = embedding.view(*original_shape[:-1], -1)
        return embedding

    @staticmethod
    def _fetch_field_inputs(batch: dict, field: str) -> None | tuple[torch.Tensor, torch.Tensor]:
        keys = [f"{field}.{key}" for key in ["input_ids", "attention_mask"]]
        if not all(key in batch for key in keys):
            return None
        return (batch[keys[0]], batch[keys[1]])

    def encode(self, batch: dict, **kws: typ.Any) -> dict[str, torch.Tensor]:
        """Computes the embeddings for the query and the document.

        NOTE: queries and documents are concatenated so representations can be obainted with a single encoder pass.
        """
        fields_data: list[tuple[str, torch.Size]] = []
        input_ids: None | torch.Tensor = None
        attention_mask: None | torch.Tensor = None

        # Collect fields_data and concatenate `input_ids`/`attention_mask`.
        for field, output_key in FIELD_MAPPING.items():
            inputs = self._fetch_field_inputs(batch, field)
            if inputs is None:
                continue

            # Unpack `input_ids`/`attention_mask` and concatenate with the other keys
            input_ids_, attention_mask_ = inputs
            fields_data.append((output_key, input_ids_.shape[:-1]))
            input_ids_ = _flatten(input_ids_, data_dim=1)
            attention_mask_ = _flatten(attention_mask_, data_dim=1)
            if input_ids is None:
                input_ids = input_ids_
                attention_mask = attention_mask_
            else:
                input_ids = torch.cat([input_ids, input_ids_], dim=0)
                attention_mask = torch.cat([attention_mask, attention_mask_], dim=0)  # type: ignore

        # Input validation
        if input_ids is None:
            raise ValueError(
                f"No fields to process. Batch keys = {batch.keys()}. Expected fields = {FIELD_MAPPING.keys()}."
            )

        # Process the inputs with the Encoder
        encoded: modeling_outputs.BaseModelOutputWithPooling = self.encoder(input_ids, attention_mask)
        embedding = encoded.pooler_output

        # Unpack the embedding
        outputs = {}
        embedding_dim = embedding.shape[1:]
        chuncks = torch.split(embedding, [math.prod(s) for _, s in fields_data])
        for (key, shape), chunk in zip(fields_data, chuncks):
            outputs[key] = chunk.view(*shape, *embedding_dim)

        return outputs

    def evaluate(
        self,
        batch: dict[str, typ.Any],
        *,
        filter_output: bool = True,
        compute_metrics: bool = True,
        **kws: typ.Any,
    ) -> dict[str, typ.Any]:  # noqa: ARG002
        """Run a forward pass, compute the gradients, compute & return the metrics."""
        fwd_output = self.forward(batch)
        grad_output = self.gradients({**batch, **fwd_output})
        if compute_metrics:
            return self._compute_metrics(batch, grad_output, filter_output=filter_output)
        return grad_output

    def _compute_metrics(
        self,
        batch: dict[str, torch.Tensor],
        grad_output: dict[str, torch.Tensor],
        filter_output: bool = True,
    ) -> dict[str, torch.Tensor]:
        output = copy.copy(grad_output)
        if self.monitor is not None:
            with torch.no_grad():
                output.update(self.monitor(output))

        # filter the output and append diagnostics
        if filter_output:
            output = _filter_model_output(output)  # type: ignore
        output.update({k: v for k, v in batch.items() if k.startswith("diagnostics.")})
        return output

    def generate(self, batch: dict[str, typ.Any], **kws: typ.Any) -> dict[str, typ.Any]:
        """Generation is not supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support generation.")


def _filter_model_output(output: dict[str, typ.Any]) -> dict[str, typ.Any]:
    def _filter_fn(key: str, _: torch.Tensor) -> bool:
        return not str(key).startswith("_")

    return {key: value for key, value in output.items() if _filter_fn(key, value)}


@hashregister(Ranker)
def _hash_ranker(hasher: Hasher, value: Ranker) -> str:  # noqa: ARG001
    return fingerprint.fingerprint_torch_module(value)


def _flatten(x: torch.Tensor, data_dim: int = 0) -> torch.Tensor:
    return x.view(-1, *x.shape[-data_dim:])
