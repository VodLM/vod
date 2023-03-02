from __future__ import annotations

import functools
from typing import Any, Optional, Union, Iterable

import pytorch_lightning as pl
import rich
import torch
import transformers
from datasets.fingerprint import hashregister, Hasher
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from torch.optim.lr_scheduler import _LRScheduler  # type: ignore
from transformers import T5EncoderModel, BertModel, BertConfig

from raffle_ds_research.ml_models.gradients import Gradients
from raffle_ds_research.ml_models.monitor import Monitor

TransformerEncoder = Union[T5EncoderModel, BertModel]


def only_trainable(parameters: Iterable[torch.nn.Parameter]):
    return (p for p in parameters if p.requires_grad)


def maybe_instantiate(conf_or_obj: Union[Any, DictConfig], **kwargs):
    if isinstance(conf_or_obj, (DictConfig, dict)):
        return instantiate(conf_or_obj, **kwargs)


class Ranker(pl.LightningModule):
    _output_size: int

    def __init__(
        self,
        encoder: TransformerEncoder,
        gradients: Gradients,
        optimizer: Optional[dict | DictConfig | functools.partial] = None,
        scheduler: Optional[dict | DictConfig | functools.partial] = None,
        monitor: Optional[Monitor] = None,
        embedding_size: Optional[int] = 512,
        use_pooler_layer: bool = False,
        init_proj_to_zero: bool = True,
    ):
        super().__init__()
        if isinstance(optimizer, (dict, DictConfig)):
            optimizer = maybe_instantiate(optimizer)
            assert isinstance(optimizer, functools.partial)
        if isinstance(scheduler, (dict, DictConfig)):
            scheduler = maybe_instantiate(scheduler)
            assert isinstance(scheduler, functools.partial)

        self.optimizer_cls: functools.partial = optimizer
        self.scheduler_cls: functools.partial = scheduler
        self.encoder = encoder
        self.gradients = gradients
        self.monitor = monitor

        # projection layer
        self.use_pooler_layer = use_pooler_layer
        h_model = self._infer_model_output_size(encoder)
        if embedding_size is None:
            self._output_size = h_model
            self.proj = torch.nn.Identity()
        else:
            self.proj = torch.nn.Linear(h_model, embedding_size, bias=False)
            self._output_size = embedding_size
            if init_proj_to_zero:
                self.proj.weight.data.zero_()

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
        # optimizer and scheduler

        # define the optimizer using the above groups
        # todo: do not apply weight decay to bias and LayerNorm parameters
        #       do this with a special constructor for the optimizer
        optimizer = self.optimizer_cls(self.parameters())

        # defile the learning rate scheduler
        lr_scheduler = self.scheduler_cls(optimizer)

        rich.print(
            {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            },
        )

        output = {
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
        return output

    def _forward_field(self, batch: dict) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        original_shape = input_ids.shape
        input_ids = input_ids.view(-1, original_shape[-1])
        attention_mask = attention_mask.view(-1, original_shape[-1])
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        if self.use_pooler_layer:
            embedding = outputs.pooler_output
        else:
            embedding = outputs.last_hidden_state[..., 0, :]
        embedding = self.proj(embedding)
        embedding = embedding.view(*original_shape[:-1], -1)
        return embedding

    @staticmethod
    def _fetch_fields(batch: dict, field: str) -> dict[str, torch.Tensor]:
        keys = ["input_ids", "attention_mask"]
        output = {key: batch[f"{field}.{key}"] for key in keys}
        return output

    def forward(self, batch: dict, **kwargs: Any) -> dict:
        mapping = {"question": "hq", "section": "hd"}
        output = {}
        for field, key in mapping.items():
            output[key] = self._forward_field(self._fetch_fields(batch, field))
        if len(output) == 0:
            raise ValueError(
                f"No fields to process. " f"Batch keys = {batch.keys()}. " f"Expected fields = {mapping.keys()}."
            )

        output = self.gradients({**batch, **output})
        output.update(self._input_stats(batch))
        return output

    @staticmethod
    def _input_stats(batch: dict) -> dict:
        output = {}
        keys = ["question.input_ids", "section.input_ids"]
        for key in keys:
            try:
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    output[f"{key}.length"] = float(value.shape[-1])
            except KeyError:
                logger.warning(f"Key {key} not found in batch")
        return output

    def _step(self, batch: dict, batch_idx: Optional[int] = None, *, split: str, **kwargs) -> dict:
        output = self(batch)

        # todo: update metrics in _step_metrics
        if self.monitor is not None:
            self.monitor.update(output, split=split)
            if self.monitor.on_step(split):
                metrics = self.monitor.compute(split=split)
                output.update(metrics)
                self.monitor.reset(split=split)

        # filter the output and log
        output = self._filter_output(output)
        self._log_metrics(output, split=split, on_step=True, on_epoch=False)

        return output

    @torch.no_grad()
    def _log_metrics(self, output: dict, split: str, **kwargs) -> None:
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                if value.numel() > 1:
                    value = value.mean()

            if "/" not in key:
                key = f"{split}/{key}"
            self.log(key, value, **kwargs)

    @staticmethod
    def _filter_output(output: dict) -> dict:
        def fn(key, value):
            return not str(key).startswith("_")

        output = {key: value for key, value in output.items() if fn(key, value)}

        return output

    def training_step(self, *args: Any, **kwargs: Any) -> dict:
        return self._step(*args, split="train", **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> dict:
        return self._step(*args, split="val", **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> dict:
        return self._step(*args, split="test", **kwargs)

    # Compute metrics
    def _on_epoch_end(self, split: str) -> dict:
        if self.monitor is not None:
            summary = self.monitor.compute(split=split)
            self._log_metrics(summary, split=split, prog_bar=True)
            return summary
        else:
            return {}

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end(split="train")

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(split="val")

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(split="test")

    # Init monitors
    def _on_epoch_start(self, split: str):
        if self.monitor is not None:
            self.monitor.reset(split=split)

    def on_train_epoch_start(self) -> None:
        self._on_epoch_start(split="train")

    def on_validation_epoch_start(self) -> None:
        self._on_epoch_start(split="val")

    def on_test_epoch_start(self) -> None:
        self._on_epoch_start(split="test")


@hashregister(Ranker)
def _hash_ranker(hasher: Hasher, value: Ranker):
    hasher = Hasher()
    for k, v in value.state_dict().items():
        hasher.update(k)
        u = serialize_tensor(v)
        hasher.update(u)
    return hasher.hexdigest()
