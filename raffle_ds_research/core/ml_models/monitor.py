from __future__ import annotations

import copy
import math
from abc import ABC
from typing import Any, Optional, Tuple

import hydra
import omegaconf
import torch
import torchmetrics
from torch import nn
from torchmetrics import Metric, MetricCollection

SPLIT_NAMES = ["train", "val", "test"]
MetricsBySplits = dict[str, dict[str, Metric]]


def _safe_split(split: str) -> str:
    """Return a safe split name for a torch.nn.ModuleDict to avoid conflicts with the parameter name `train`"""
    return f"_{split}"


class Monitor(nn.Module):
    """A Monitor is an object that monitors the training process by
    computing and collecting metrics"""

    _splits = list[str]
    metrics: torch.nn.ModuleDict[str, MetricCollection]
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

        if isinstance(log_on_step, omegaconf.DictConfig):
            log_on_step = omegaconf.OmegaConf.to_object(log_on_step)

        if isinstance(log_on_step, dict):
            if set(log_on_step.keys()) != set(splits):
                raise ValueError(f"Expected log_on_step to have keys {splits}, but got {list(log_on_step.keys())}")
        else:
            log_on_step = {split: log_on_step for split in splits}

        if isinstance(metrics, omegaconf.DictConfig):
            metrics = hydra.utils.instantiate(metrics)

        if isinstance(metrics, dict):
            if set(metrics.keys()) != set(splits):
                raise ValueError(f"Expected metrics to have keys {splits}, but got {list(metrics.keys())}")
        else:
            metrics = {split: copy.deepcopy(metrics) for split in splits}

        self._log_on_step = log_on_step
        self._splits = splits
        self.metrics = torch.nn.ModuleDict({_safe_split(split): metric for split, metric in metrics.items()})

    def on_step(self, split: str) -> bool:
        """Return True if the metrics should be computed and logged on step"""
        if split not in self._log_on_step.keys():
            raise TypeError(f"Unknown split type: {type(split)}. Expected one of {self._log_on_step.keys()}.")
        return self._log_on_step[split]

    def forward(self, data: dict, split: str) -> dict:
        """Compute the metrics"""
        metric = self.metrics[_safe_split(split)]
        args = self._make_args(data)
        return metric.forward(*args)

    def reset(self, split: Optional[str | list[str]] = None):
        """Reset the metrics"""
        splits = self._get_splits_arg(split)
        for split in splits:
            self.metrics[_safe_split(split)].reset()

        return self

    def _get_splits_arg(self, split):
        if split is None:
            splits = self._splits
        elif isinstance(split, str):
            splits = [split]
        elif isinstance(split, list):
            splits = split
        else:
            raise TypeError(f"Unknown split type: {type(split)}")

        if not set(splits).issubset(self._splits):
            raise ValueError(f"Unknown split: {split}. Expected one of {self._splits}.")

        return splits

    def update_from_retrieval_batch(self, batch: dict[str, Any], field: str = "section"):
        """Update the metrics given a raw `retrieval` batch"""
        for key in self._splits:
            targets: torch.Tensor = batch[f"{field}.label"]
            try:
                scores: torch.Tensor = batch[f"{field}.{key}"]
            except KeyError:
                continue
            # set the retrieval to -inf when the score is undefined
            scores = scores.masked_fill(torch.isnan(scores), -math.inf)
            args = self._make_args({"_logits": scores, "_targets": targets})
            self.metrics[_safe_split(key)].update(*args)

    @torch.no_grad()
    def update(self, data: dict, split: str):
        """Update the metrics"""
        args = self._make_args(data)
        self.metrics[_safe_split(split)].update(*args)

    @torch.no_grad()
    def compute(self, split: Optional[str] = None, prefix: str = "") -> dict[str, torch.Tensor]:
        """Compute the metrics. Wrap with try/except to avoid raising exception when there are no
        metrics to compute."""
        if split is None:
            metrics = {}
            for split in self._splits:
                metrics.update(self.compute(split=split, prefix=prefix))
            return metrics

        if isinstance(self.metrics[_safe_split(split)], MetricCollection):
            metrics = {}
            for key, metric in self.metrics[_safe_split(split)].items():
                try:
                    metrics[key] = metric.compute()
                except ValueError:
                    ...
        else:
            metrics = self.metrics[_safe_split(split)].compute()

        metrics = {f"{prefix}{split}/{key}": value for key, value in metrics.items()}
        return metrics

    @staticmethod
    def _make_args(data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        preds: torch.Tensor = data["_logits"]
        targets: torch.Tensor = data["_targets"]
        return preds, targets


def _rank_labels(*, labels: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    sort_ids = torch.argsort(scores, dim=-1, descending=True)
    ranked_labels = torch.gather(labels, dim=-1, index=sort_ids)
    return ranked_labels


def _mask_inputs(preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = preds.isnan()
    preds = preds.masked_fill(mask, -math.inf)
    mask = mask | preds.isinf()
    target = target.masked_fill(mask, 0)
    return preds, target


class HitRate(torchmetrics.Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, top_k: int, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.add_state("hits", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds, target = _mask_inputs(preds, target)
        ranked_labels = _rank_labels(labels=target, scores=preds)
        ranked_labels = ranked_labels[..., : self.top_k]
        hits = (ranked_labels > 0).any(dim=-1)
        self.hits += hits.sum().to(self.hits)
        self.total += hits.numel()

    def compute(self) -> torch.Tensor:
        return self.hits.float() / self.total


class AveragedMetric(torchmetrics.Metric, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _update(self, values: torch.Tensor) -> None:
        self.value += values.sum().to(self.value)
        self.weight += values.numel()

    def compute(self) -> torch.Tensor:
        return self.value / self.weight


class MeanReciprocalRank(AveragedMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds, target = _mask_inputs(preds, target)
        ranked_labels = _rank_labels(labels=target, scores=preds)
        idx_first_non_zero = _arg_first_non_zero(ranked_labels)
        has_positive = (ranked_labels > 0).sum(dim=-1) > 0
        mrr = 1.0 / (1 + idx_first_non_zero)
        mmr = torch.where(has_positive, mrr, 0)
        self._update(mmr)


def _arg_first_non_zero(values: torch.Tensor) -> torch.Tensor:
    ids = torch.arange(values.shape[-1], device=values.device)
    nnz_ordered_values = torch.where(values > 0, ids, 1 + ids.max())
    idx_first_non_zero = nnz_ordered_values.argmin(dim=-1)
    return idx_first_non_zero


class NormalizedDCG(AveragedMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, top_k: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds, target = _mask_inputs(preds, target)
        batch_size = preds[..., 0].numel()
        ndcg = torchmetrics.functional.retrieval_normalized_dcg(preds=preds, target=target, k=self.top_k)
        # `retrieval_normalized_dcg` returns the mean over the batch,
        # so we need to multiply by the batch size to get the sum
        ndcg = ndcg.repeat(batch_size)
        self._update(ndcg)


def retrieval_metric_factory(name: str, **kwargs) -> Metric:
    """Instantiate a torchmetrics retrieval metric from a string name."""
    if "@" in name:
        name, k = name.split("@")
        top_k = int(k)
    else:
        top_k = None

    avail_cls = {
        "mrr": MeanReciprocalRank,
        "ndcg": NormalizedDCG,
        "hitrate": HitRate,
    }

    cls = avail_cls[name]

    return cls(top_k=top_k, **kwargs)


class RetrievalMetricCollection(MetricCollection):
    def __init__(self, metrics: list[str], **kwargs):
        def clean_name(x: str) -> str:
            x = x.replace("@", "_")
            return x

        metrics = {clean_name(name): retrieval_metric_factory(name=name, **kwargs) for name in metrics}
        super().__init__(metrics=metrics)
