# pylint: disable=too-many-arguments,arguments-differ,inconsistent-return-statements,too-many-instance-attributes,fixme
from __future__ import annotations

import functools
import re
from typing import Any, Iterable, Optional, Union

import lightning.pytorch as pl
import loguru
import torch
import transformers
from datasets.fingerprint import Hasher, hashregister
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from transformers import BertConfig, BertModel, T5EncoderModel

from raffle_ds_research.core.ml_models.gradients import Gradients
from raffle_ds_research.core.ml_models.monitor import Monitor
from raffle_ds_research.tools.pipes import fingerprint_torch_module

TransformerEncoder = Union[T5EncoderModel, BertModel]


def only_trainable(parameters: Iterable[torch.nn.Parameter]) -> Iterable[torch.nn.Parameter]:
    """Filter out parameters that do not require gradients."""
    return (p for p in parameters if p.requires_grad)


def maybe_instantiate(conf_or_obj: Union[Any, DictConfig], **kwargs: Any) -> object:
    """Instantiate a config if needed."""
    if isinstance(conf_or_obj, (DictConfig, dict)):
        return instantiate(conf_or_obj, **kwargs)
    return None


PBAR_MATCH_PATTERN = re.compile(r"(loss|ndcg|mrr)$")


class Ranker(pl.LightningModule):
    """Deep ranking model using a Transformer encoder as a backbone."""

    _output_size: int

    def __init__(  # noqa: PLR0913
        self,
        encoder: TransformerEncoder,
        gradients: Gradients,
        optimizer: Optional[dict | DictConfig | functools.partial] = None,
        scheduler: Optional[dict | DictConfig | functools.partial] = None,
        monitor: Optional[Monitor] = None,
        embedding_size: Optional[int] = 512,
        use_pooler_layer: bool = False,
        compile: bool = False,
    ):
        super().__init__()
        if isinstance(optimizer, (dict, DictConfig)):
            optimizer = maybe_instantiate(optimizer)
            if not isinstance(optimizer, functools.partial):
                raise TypeError(f"Expected a partial function, got {type(optimizer)}")
        if isinstance(scheduler, (dict, DictConfig)):
            scheduler = maybe_instantiate(scheduler)
            if not isinstance(scheduler, functools.partial):
                raise TypeError(f"Expected a partial function, got {type(scheduler)}")

        self.optimizer_cls: functools.partial = optimizer
        self.scheduler_cls: functools.partial = scheduler
        self.gradients = gradients
        self.monitor = monitor
        self.encoder = encoder

        # projection layer
        self.use_pooler_layer = use_pooler_layer
        h_model = self._infer_model_output_size(encoder)
        if embedding_size is None:
            self._output_size = h_model
            self.proj = torch.nn.Identity()
        else:
            self.proj = torch.nn.Linear(h_model, embedding_size, bias=False)
            self._output_size = embedding_size

        # setup the encoder
        if compile:
            loguru.logger.info("Compiling the encoder...")
            self.encoder = torch.compile(self.encoder)
            # loguru.logger.info("Compiling the gradients...")
            # self.gradients = torch.compile(self.gradients)

    def get_output_shape(self, model_output_key: Optional[str] = None) -> tuple[int, ...]:  # noqa: ARG002
        """Dimension of the model output."""
        return (self._output_size,)

    @staticmethod
    def _infer_model_output_size(encoder: TransformerEncoder) -> int:
        if isinstance(encoder.config, BertConfig):
            h_model = encoder.config.hidden_size
        elif isinstance(encoder.config, transformers.T5Config):
            h_model = encoder.config.d_model
        else:
            raise ValueError(f"Unknown encoder config type: {type(encoder.config)}")
        return h_model

    def configure_optimizers(self) -> dict:
        """Configure the optimizer and the learning rate scheduler."""
        # define the optimizer using the above groups
        # todo: do not apply weight decay to bias and LayerNorm parameters
        #       do this with a special constructor for the optimizer
        optimizer = self.optimizer_cls(self.parameters())

        # defile the learning rate scheduler
        lr_scheduler = self.scheduler_cls(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after an optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            },
        }

    def _forward_field(self, batch: dict) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        original_shape = input_ids.shape
        input_ids = input_ids.view(-1, original_shape[-1])
        attention_mask = attention_mask.view(-1, original_shape[-1])
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        embedding = outputs.pooler_output if self.use_pooler_layer else outputs.last_hidden_state[..., 0, :]
        embedding = self.proj(embedding)
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
        """Forward pass through the model. Only computes the embeddings for the query and the document."""
        return self.predict(batch)

    def predict(self, batch: dict, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Computes the embeddings for the query and the document."""
        mapping = {"question": "hq", "section": "hd"}
        output = {}
        for field, key in mapping.items():
            fields = self._fetch_fields(batch, field)
            if fields is None:
                continue
            output[key] = self._forward_field(fields)
        if len(output) == 0:
            raise ValueError(
                f"No fields to process. " f"Batch keys = {batch.keys()}. Expected fields = {mapping.keys()}."
            )
        return output

    def _step(self, batch: dict[str, Any], *, split: str, **kwargs: Any) -> dict[str, Any]:
        output = self.forward(batch)
        output = self.gradients({**batch, **output})
        output.update(_compute_input_stats(batch, prefix=f"{split}/"))

        if self.monitor is not None:
            if self.monitor.on_step(split):
                output.update(self.monitor.forward(output, split=split))
            else:
                self.monitor.update(output, split=split)

        # filter the output and log
        output = self._filter_output(output)
        on_step = split == "train"
        self._log_metrics(output, split=split, on_step=on_step, on_epoch=not on_step)

        return output

    @torch.no_grad()
    def _log_metrics(
        self,
        output: dict[str, Any],
        split: str,
        prog_bar: Optional[bool] = None,
        on_epoch: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                value = value.mean()  # noqa: PLW2901

            if "/" not in key:
                key = f"{split}/{key}"  # noqa: PLW2901

            if prog_bar is None:
                prog_bar = PBAR_MATCH_PATTERN.search(key) is not None
            if on_epoch is True:
                kwargs = {**kwargs, "sync_dist": True}
            self.log(key, value, prog_bar=prog_bar, on_epoch=on_epoch, **kwargs)

    @staticmethod
    def _filter_output(output: dict[str, Any]) -> dict[str, Any]:
        def _filter_fn(key: str, _: torch.Tensor) -> bool:
            return not str(key).startswith("_")

        return {key: value for key, value in output.items() if _filter_fn(key, value)}

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Implements the lightning training step."""
        return self._step(*args, split="train", **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Implements the lightning validation step."""
        return self._step(*args, split="val", **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Implements the lightning test step."""
        return self._step(*args, split="test", **kwargs)

    # Compute metrics
    def _on_epoch_end(self, split: str) -> dict[str, Any]:
        if self.monitor is not None and not self.monitor.on_step(split):
            summary = self.monitor.compute(split=split)
            self._log_metrics(summary, split=split, on_epoch=True)
            return summary

        return {}

    def on_train_epoch_end(self) -> None:
        """Called at the end of the train epoch."""
        self._on_epoch_end(split="train")

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch."""
        self._on_epoch_end(split="val")

    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch."""
        self._on_epoch_end(split="test")

    # Init monitors
    def _on_epoch_start(self, split: str) -> None:
        if self.monitor is not None:
            self.monitor.reset(split=split)

    def on_train_epoch_start(self) -> None:
        """Called at the beginning of the train epoch."""
        self._on_epoch_start(split="train")

    def on_validation_epoch_start(self) -> None:
        """Called at the beginning of the validation epoch."""
        self._on_epoch_start(split="val")

    def on_test_epoch_start(self) -> None:
        """Called at the beginning of the test epoch."""
        self._on_epoch_start(split="test")


def _compute_input_stats(batch: dict, prefix: str = "") -> dict[str, float]:
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
                output[f"{prefix}{log_key}_length"] = float(shp[-1])
                if len(shp) > 2:  # noqa: PLR2004
                    output[f"{prefix}{log_key}_n"] = float(shp[-2])
        except KeyError:
            logger.warning(f"Key {key} not found in batch")
    return output


@hashregister(Ranker)
def _hash_ranker(hasher: Hasher, value: Ranker) -> str:
    return fingerprint_torch_module(hasher, value)
