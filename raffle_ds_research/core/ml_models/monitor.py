from __future__ import annotations

import copy
import math
from typing import Any, Optional, Tuple

import hydra
import omegaconf
import torch
import torchmetrics
from torch import nn
from torchmetrics import Metric, MetricCollection

from raffle_ds_research.tools import pipes

SPLIT_NAMES = ["train", "val", "test"]
MetricsBySplits = dict[str, dict[str, Metric]]


class Monitor(nn.Module):
    """A Monitor is an object that monitors the training process by
    computing and collecting metrics"""

    metrics: dict[str, MetricCollection]
    _log_on_step: dict[str, bool]

    def __init__(
        self,
        metrics: MetricCollection | dict[str, MetricCollection],
        splits: list[str] = None,
        log_on_step: bool | dict[str, bool] = False,
    ):
        super().__init__()
        if splits is None:
            splits = copy.copy(SPLIT_NAMES)

        if isinstance(log_on_step, dict):
            assert set(log_on_step.keys()) == set(SPLIT_NAMES)
        else:
            log_on_step = {split: log_on_step for split in splits}

        if isinstance(metrics, omegaconf.DictConfig):
            metrics = hydra.utils.instantiate(metrics)

        if isinstance(metrics, dict):
            assert set(metrics.keys()) == set(SPLIT_NAMES)
        else:
            metrics = {split: copy.deepcopy(metrics) for split in splits}

        self._log_on_step = log_on_step
        self.metrics = metrics

    def log_on_step(self, split: str) -> bool:
        """Return True if the metrics should be logged on step"""
        if split not in SPLIT_NAMES:
            raise TypeError(f"Unknown split type: {type(split)}")
        return self._log_on_step[split]

    def forward(self, data: dict, split: str) -> dict:
        """Compute the metrics"""
        metric = self.metrics[split]
        args = self._make_args(data)
        return metric.forward(*args)

    def reset(self, split: Optional[str | list[str]] = None):
        """Reset the metrics"""
        splits = self._get_splits_arg(split)
        for split in splits:
            self.metrics[split].reset()

        return self

    def _get_splits_arg(self, split):
        if split is None:
            splits = list(self.metrics.keys())
        elif isinstance(split, str):
            splits = [split]
        elif isinstance(split, list):
            splits = split
        else:
            raise TypeError(f"Unknown split type: {type(split)}")

        if not set(splits).issubset(self.metrics.keys()):
            raise ValueError(f"Unknown split: {split}. Expected one of {self.metrics.keys()}.")

        return splits

    def update_from_retrieval_batch(self, batch: dict[str, Any], field: str = "section"):
        """Update the metrics given a raw `retrieval` batch"""
        for key in self.metrics:
            targets: torch.Tensor = batch[f"{field}.label"]
            try:
                scores: torch.Tensor = batch[f"{field}.{key}"]
            except KeyError:
                continue
            # set the retrieval to -inf when the score is undefined
            scores = scores.masked_fill(torch.isnan(scores), -math.inf)
            args = self._make_args({"_logits": scores, "_targets": targets})
            self.metrics[key].update(*args)

    @torch.no_grad()
    def update(self, data: dict, split: str):
        """Update the metrics"""
        args = self._make_args(data)
        self.metrics[split].update(*args)

    @torch.no_grad()
    def compute(self, split: Optional[str] = None, prefix: str = "") -> dict[str, torch.Tensor]:
        """Compute the metrics. Wrap with try/except to avoid raising exception when there are no
        metrics to compute."""
        if split is None:
            metrics = {}
            for split in self.metrics:
                metrics.update(self.compute(split=split, prefix=prefix))
            return metrics

        if isinstance(self.metrics[split], MetricCollection):
            metrics = {}
            for key, metric in self.metrics[split].items():
                try:
                    metrics[key] = metric.compute()
                except ValueError:
                    ...
        else:
            metrics = self.metrics[split].compute()

        metrics = {f"{prefix}{split}/{key}": value for key, value in metrics.items()}
        return metrics

    @staticmethod
    def _make_args(data: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds: torch.Tensor = data["_logits"]
        targets: torch.Tensor = data["_targets"]

        # todo: remove
        sort_ids = torch.argsort(preds, dim=-1, descending=True)
        preds = preds.gather(-1, sort_ids)
        targets = targets.gather(-1, sort_ids)

        indices = torch.arange(len(preds), device=preds.device)
        indices = indices.unsqueeze(-1).expand(-1, preds.shape[-1])
        return preds, targets, indices


def retrieval_metric_factory(name: str, **kwargs) -> Metric:
    """Instantiate a torchmetrics retrieval metric from a string name."""
    if "@" in name:
        name, k = name.split("@")
        k = int(k)
    else:
        k = None

    avail_cls = {
        "mrr": torchmetrics.RetrievalMRR,
        "map": torchmetrics.RetrievalMAP,
        "ndcg": torchmetrics.RetrievalNormalizedDCG,
        "p": torchmetrics.RetrievalPrecision,
        "r": torchmetrics.RetrievalRecall,
        "hitrate": torchmetrics.RetrievalHitRate,
    }

    cls = avail_cls[name]

    return cls(k=k, **kwargs)


class RetrievalMetricCollection(MetricCollection):
    def __init__(self, metrics: list[str], **kwargs):
        def clean_name(x: str) -> str:
            x = x.replace("@", "_")
            return x

        metrics = {clean_name(name): retrieval_metric_factory(name=name, **kwargs) for name in metrics}
        super().__init__(metrics=metrics)
