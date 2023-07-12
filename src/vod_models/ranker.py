from __future__ import annotations

import copy
import functools
from typing import Any, Optional, Union

import torch
from datasets.fingerprint import Hasher, hashregister
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from raffle_ds_research.core.ml.gradients import Gradients
from raffle_ds_research.core.ml.monitor import RetrievalMonitor
from raffle_ds_research.tools import interfaces
from raffle_ds_research.tools.pipes import fingerprint_torch_module
from transformers import pytorch_utils, trainer_pt_utils  # type: ignore


def _maybe_instantiate(conf_or_obj: Union[Any, DictConfig], **kwargs: Any) -> object:
    """Instantiate a config if needed."""
    if isinstance(conf_or_obj, (DictConfig, dict)):
        return instantiate(conf_or_obj, **kwargs)
    return None


class Ranker(torch.nn.Module):
    """Deep ranking model using a Transformer encoder as a backbone."""

    _output_size: int
    encoder: interfaces.ProtocolEncoder

    def __init__(  # noqa: PLR0913
        self,
        encoder: interfaces.ProtocolEncoder,
        gradients: Gradients,
        optimizer: Optional[dict | DictConfig | functools.partial] = None,
        scheduler: Optional[dict | DictConfig | functools.partial] = None,
        monitor: Optional[RetrievalMonitor] = None,
        compile_encoder: bool = False,
        compile_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if isinstance(optimizer, (dict, DictConfig)):
            optimizer = _maybe_instantiate(optimizer)  # type: ignore
            if not isinstance(optimizer, functools.partial):
                raise TypeError(f"Expected a partial function, got {type(optimizer)}")
        if isinstance(scheduler, (dict, DictConfig)):
            scheduler = _maybe_instantiate(scheduler)  # type: ignore
            if not isinstance(scheduler, functools.partial):
                raise TypeError(f"Expected a partial function, got {type(scheduler)}")

        self.optimizer_cls: functools.partial = optimizer  # type: ignore
        self.scheduler_cls: functools.partial = scheduler  # type: ignore
        self.gradients = gradients
        self.monitor = monitor

        # compile the encoder
        if compile_encoder:
            logger.info("Compiling the encoder..")
            encoder = torch.compile(
                encoder,
                **(compile_kwargs or {}),
            )  # type: ignore

        self.encoder = encoder

    def get_output_shape(self, model_output_key: Optional[str] = None) -> tuple[int, ...]:  # noqa: ARG002
        """Dimension of the model output."""
        return self.encoder.get_output_shape(model_output_key)  # type: ignore

    def get_optimizer(self, module: Optional[torch.nn.Module] = None) -> torch.optim.Optimizer:
        """Configure the optimizer and the learning rate scheduler."""
        opt_model = module or self

        # Fetch the weight decay from the optimizer
        if isinstance(self.optimizer_cls, functools.partial):
            weight_decay = self.optimizer_cls.keywords.get("weight_decay", None)
        else:
            weight_decay = None

        # If no weight_decay is provided, instantiate the optimizer directly
        if weight_decay is None:
            return self.optimizer_cls(opt_model.parameters())

        # Instantiate the optimizer à la HuggingFace
        # https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/trainer.py#L1075C15-L1075C15
        decay_parameters = trainer_pt_utils.get_parameter_names(opt_model, pytorch_utils.ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return self.optimizer_cls(optimizer_grouped_parameters)

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> None | torch.optim.lr_scheduler._LRScheduler:
        """Init the learning rate scheduler."""
        if self.scheduler_cls is None:
            return None

        return self.scheduler_cls(optimizer)

    def _forward_field(self, batch: dict, field: Optional[interfaces.FieldType] = None) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        original_shape = input_ids.shape
        input_ids = input_ids.view(-1, original_shape[-1])
        attention_mask = attention_mask.view(-1, original_shape[-1])
        embedding = self.encoder(input_ids, attention_mask, field=field)
        embedding = embedding.view(*original_shape[:-1], -1)
        return embedding

    @staticmethod
    def _fetch_fields(batch: dict, field: str) -> Optional[dict[str, torch.Tensor]]:
        keys = ["input_ids", "attention_mask"]
        keys_map = {f"{field}.{key}": key for key in keys}
        if not all(key in batch for key in keys_map):
            return None
        return {key_to: batch[key_from] for key_from, key_to in keys_map.items()}

    def encode(self, batch: dict, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Computes the embeddings for the query and the document."""
        mapping = {"question": "hq", "section": "hd"}
        output = {}
        for field, key in mapping.items():
            fields = self._fetch_fields(batch, field)
            if fields is None:
                continue
            output[key] = self._forward_field(fields, interfaces.FieldType(field))
        if len(output) == 0:
            raise ValueError(
                f"No fields to process. " f"Batch keys = {batch.keys()}. Expected fields = {mapping.keys()}."
            )

        return output

    def forward(self, batch: dict, *, mode: str = "encode", **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        if mode == "encode":
            return self.encode(batch, **kwargs)
        if mode == "evaluate":
            return self.evaluate(batch, **kwargs)

        raise ValueError(f"Unknown mode {mode}.")

    def evaluate(
        self,
        batch: dict[str, Any],
        *,
        filter_output: bool = True,
        compute_metrics: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:  # noqa: ARG002
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


def _filter_model_output(output: dict[str, Any]) -> dict[str, Any]:
    def _filter_fn(key: str, _: torch.Tensor) -> bool:
        return not str(key).startswith("_")

    return {key: value for key, value in output.items() if _filter_fn(key, value)}


@hashregister(Ranker)
def _hash_ranker(hasher: Hasher, value: Ranker) -> str:
    return fingerprint_torch_module(hasher, value)
