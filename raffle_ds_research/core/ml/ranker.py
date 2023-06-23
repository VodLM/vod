from __future__ import annotations

import functools
from typing import Any, Iterable, Optional, Union

import loguru
import torch
from datasets.fingerprint import Hasher, hashregister
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from raffle_ds_research.core.ml.gradients import Gradients
from raffle_ds_research.core.ml.monitor import RetrievalMonitor
from raffle_ds_research.tools import interfaces
from raffle_ds_research.tools.pipes import fingerprint_torch_module


def _only_trainable(parameters: Iterable[torch.nn.Parameter]) -> Iterable[torch.nn.Parameter]:
    """Filter out parameters that do not require gradients."""
    return (p for p in parameters if p.requires_grad)


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
        compile: bool = False,
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
        self.encoder = encoder

        # setup the encoder
        if compile:
            loguru.logger.info("Compiling the encoder...")
            self.encoder = torch.compile(self.encoder)  # type: ignore
            # loguru.logger.info("Compiling the gradients...")
            # self.gradients = torch.compile(self.gradients)

    def get_output_shape(self, model_output_key: Optional[str] = None) -> tuple[int, ...]:  # noqa: ARG002
        """Dimension of the model output."""
        return self.encoder.get_output_shape(model_output_key)  # type: ignore

    def get_optimizer(self, module: Optional[torch.nn.Module] = None) -> torch.optim.Optimizer:
        """Configure the optimizer and the learning rate scheduler."""
        if module is None:
            module = self
        return self.optimizer_cls(module.parameters())

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> None | torch.optim.lr_scheduler._LRScheduler:
        """Init the learning rate scheduler."""
        if self.scheduler_cls is None:
            return None

        return self.scheduler_cls(optimizer)

    def _forward_field(self, batch: dict, field: str) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        original_shape = input_ids.shape
        input_ids = input_ids.view(-1, original_shape[-1])
        attention_mask = attention_mask.view(-1, original_shape[-1])
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "field": field}
        embedding = self.encoder(inputs)
        embedding = embedding.view(*original_shape[:-1], -1)
        return embedding

    @staticmethod
    def _fetch_fields(batch: dict, field: str) -> Optional[dict[str, torch.Tensor]]:
        keys = ["input_ids", "attention_mask"]
        keys_map = {f"{field}.{key}": key for key in keys}
        if not all(key in batch for key in keys_map):
            return None
        return {key_to: batch[key_from] for key_from, key_to in keys_map.items()}

    def forward(self, batch: dict, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Computes the embeddings for the query and the document."""
        mapping = {"question": "hq", "section": "hd"}
        output = {}
        for field, key in mapping.items():
            fields = self._fetch_fields(batch, field)
            if fields is None:
                continue
            output[key] = self._forward_field(fields, field)
        if len(output) == 0:
            raise ValueError(
                f"No fields to process. " f"Batch keys = {batch.keys()}. Expected fields = {mapping.keys()}."
            )
        return output

    def _step(self, batch: dict[str, Any], *, split: str, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        output = self.forward(batch)
        output = self.gradients({**batch, **output})
        output.update(_compute_input_stats(batch))
        if self.monitor is not None:
            output.update(self.monitor(output))

        # filter the output and append diagnostics
        output = self._filter_output(output)  # type: ignore
        output.update({k: v for k, v in batch.items() if k.startswith("diagnostics.")})

        return output

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Implements the lightning training step."""
        return self._step(*args, split="train", **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Implements the lightning validation step."""
        return self._step(*args, split="val", **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Implements the lightning test step."""
        return self._step(*args, split="test", **kwargs)

    @staticmethod
    def _filter_output(output: dict[str, Any]) -> dict[str, Any]:
        def _filter_fn(key: str, _: torch.Tensor) -> bool:
            return not str(key).startswith("_")

        return {key: value for key, value in output.items() if _filter_fn(key, value)}


def _compute_input_stats(batch: dict) -> dict[str, float]:
    output = {}
    keys = {
        "question.input_ids": "question/input_ids",
        "section.input_ids": "section/input_ids",
    }
    for key, log_key in keys.items():
        try:
            value = batch[key]
            if isinstance(value, torch.Tensor):
                shp = value.shape
                output[f"{log_key}_length"] = float(shp[-1])
                if len(shp) > 2:  # noqa: PLR2004
                    output[f"{log_key}_n"] = float(shp[-2])
        except KeyError:
            logger.warning(f"Key {key} not found in batch")
    return output


@hashregister(Ranker)
def _hash_ranker(hasher: Hasher, value: Ranker) -> str:
    return fingerprint_torch_module(hasher, value)
