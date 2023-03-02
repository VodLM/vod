from __future__ import annotations

from typing import Optional, Tuple, Literal

import torch
import torchmetrics
from omegaconf import OmegaConf, DictConfig
from torch import nn
from torchmetrics import MetricCollection, Metric

SplitName = Literal["train", "val", "test"]
MetricsBySplits = dict[SplitName, dict[str, Metric]]


class Monitor(nn.Module):
    """A Monitor is an object that monitors the training process by
    computing and collecting metrics"""

    metrics: dict[str, MetricCollection]
    _on_step: dict[SplitName, bool]

    def __init__(self, metrics: MetricsBySplits, on_step: dict[SplitName, bool] = None):
        super().__init__()
        self.metrics, self._on_step = {}, {}
        for key, metric in metrics.items():
            metric = self._handle_metric_init(metric)
            self.metrics[key] = metric
            if on_step is None:
                self._on_step[key] = False
            else:
                self._on_step[key] = on_step[key]

    @staticmethod
    def _handle_metric_init(metric):
        if isinstance(
            metric,
            (
                dict,
                DictConfig,
            ),
        ):
            if isinstance(metric, DictConfig):
                metric = OmegaConf.to_container(metric)
            metric = MetricCollection(metric)
        elif isinstance(
            metric,
            (
                Metric,
                MetricCollection,
                RetrievalMetricCollection,
            ),
        ):
            pass
        else:
            raise TypeError(f"Unknown metric type: {type(metric)}")
        return metric

    def on_step(self, split: SplitName) -> bool:
        return self._on_step[split]

    def forward(self, data: dict, split: SplitName) -> dict:
        """Compute the metrics"""
        metric = self.metrics[split]
        args = self._make_args(data)
        return metric.forward(*args)

    def reset(self, split: Optional[str | list[str]] = None):
        """Reset the metrics"""
        splits = self._get_splits_arg(split)
        for split in splits:
            self.metrics[split].reset()

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

    @torch.no_grad()
    def update(self, data: dict, split: str):
        """Update the metrics"""
        args = self._make_args(data)
        self.metrics[split].update(*args)

    @torch.no_grad()
    def compute(self, split: str) -> dict[str, torch.Tensor]:
        """Compute the metrics"""
        metrics = self.metrics[split].compute()
        metrics = {f"{split}/{key}": value for key, value in metrics.items()}
        return metrics

    @staticmethod
    def _make_args(data: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds: torch.Tensor = data["_logits"]
        targets: torch.Tensor = data["_targets"]
        indices = torch.arange(len(preds), device=preds.device)
        indices = indices.unsqueeze(-1).expand(-1, preds.shape[-1])
        return (preds, targets, indices)


def retrieval_metric_factory(name: str) -> Metric:
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

    return cls(top_k=k)


class RetrievalMetricCollection(MetricCollection):
    def __init__(self, metrics: list[str]):
        def clean_name(x: str) -> str:
            x = x.replace("@", "_")
            return x

        metrics = {clean_name(name): retrieval_metric_factory(name=name) for name in metrics}
        super().__init__(metrics=metrics)
