# pylint: disable=no-member,arguments-differ

from __future__ import annotations

import abc
import copy
import math
from abc import ABC
from typing import Any, Iterable, Optional, Tuple

import hydra
import omegaconf
import torch
import torchmetrics
from torch import nn
from torchmetrics import Metric, MetricCollection

SPLIT_NAMES = ["train", "val", "test"]
MetricsBySplits = dict[str, dict[str, Metric]]


def _safe_split(split: str) -> str:
    """Return a safe split name for a torch.nn.ModuleDict to avoid conflicts with the parameter name `train`."""
    return f"_{split}"


class Monitor(nn.Module):
    """A Monitor is an object that monitors the training process by computing and collecting metrics."""

    _splits = list[str]
    metrics: torch.nn.ModuleDict
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
        """Return True if the metrics should be computed and logged on step."""
        if split not in self._log_on_step.keys():
            raise TypeError(f"Unknown split type: {type(split)}. Expected one of {self._log_on_step.keys()}.")
        return self._log_on_step[split]

    def forward(self, data: dict, split: str) -> dict[str, Any]:
        """Compute the metrics."""
        metric = self.metrics[_safe_split(split)]
        args = self._make_args(data)
        return metric.forward(*args)

    def reset(self, split: Optional[str | list[str]] = None) -> "Monitor":
        """Reset the metrics."""
        splits = self._get_splits_arg(split)
        for split in splits:
            self.metrics[_safe_split(split)].reset()

        return self

    def _get_splits_arg(self, split: str) -> list[str]:
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

    def update_from_retrieval_batch(self, batch: dict[str, Any], field: str = "section") -> None:
        """Update the metrics given a raw `retrieval` batch."""
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
    def update(self, data: dict, split: str) -> None:
        """Update the metrics."""
        args = self._make_args(data)
        self.metrics[_safe_split(split)].update(*args)

    @torch.no_grad()
    def compute(self, split: Optional[str] = None, prefix: str = "") -> dict[str, torch.Tensor]:
        """Compute the metrics. Wrap with try/except to avoid raising exception when there are no metrics to compute."""
        if split is None:
            metrics = {}
            for split_ in self._splits:
                metrics.update(self.compute(split=split_, prefix=prefix))
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
    return torch.gather(labels, dim=-1, index=sort_ids)


def _mask_inputs(preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = preds.isnan()
    preds = preds.masked_fill(mask, -math.inf)
    mask = mask | preds.isinf()
    target = target.masked_fill(mask, 0)
    return preds, target


class HitRate(torchmetrics.Metric):
    """Hit rate metric: same as `Accuracy`."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, top_k: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.add_state("hits", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric with the given predictions and targets."""
        preds, target = _mask_inputs(preds, target)
        ranked_labels = _rank_labels(labels=target, scores=preds)
        ranked_labels = ranked_labels[..., : self.top_k]
        hits = (ranked_labels > 0).any(dim=-1)
        self.hits += hits.sum().to(self.hits)
        self.total += hits.numel()

    def compute(self) -> torch.Tensor:
        """Compute the metric value."""
        return self.hits.float() / self.total


class AveragedMetric(torchmetrics.Metric, ABC):
    """Base class for metrics that are computed at the sample level and averaged over the entire dataset."""

    def __init__(self, top_k: Optional[int] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _update(self, values: torch.Tensor) -> None:
        self.value += values.sum().to(self.value)
        self.weight += values.numel()

    def compute(self) -> torch.Tensor:
        """Compute the metric value."""
        return self.value / self.weight


class MeanReciprocalRank(AveragedMetric):
    """Mean Reciprocal Rank (MRR) metric."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric given the predictions and the targets."""
        preds, target = _mask_inputs(preds, target)
        ranked_labels = _rank_labels(labels=target, scores=preds)
        ranked_labels = ranked_labels[..., : self.top_k]
        idx_first_non_zero = _arg_first_non_zero(ranked_labels)
        at_least_one_target = (ranked_labels > 0).sum(dim=-1) > 0
        mrr = 1.0 / (1 + idx_first_non_zero)
        mmr = torch.where(at_least_one_target, mrr, 0)
        self._update(mmr)


def _arg_first_non_zero(values: torch.Tensor) -> torch.Tensor:
    ids = torch.arange(values.shape[-1], device=values.device)
    nnz_ordered_values = torch.where(values > 0, ids, 1 + ids.max())
    return nnz_ordered_values.argmin(dim=-1)


class LightningAveragedMetric(AveragedMetric):
    """Base class for metrics that are computed at the sample level and averaged over the entire dataset."""

    # torchmetrics stuff
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric."""
        preds, target = _mask_inputs(preds, target)
        batch_size = preds[..., 0].numel()
        batch_avg = self._metric_fn(preds=preds, target=target, k=self.top_k)
        # `metric_fn` returns the mean over the batch,
        # so we need to multiply by the batch size to get the sum
        batch_avg = batch_avg.repeat(batch_size)
        self._update(batch_avg)

    @abc.abstractmethod
    def _metric_fn(self, preds: torch.Tensor, target: torch.Tensor, k: Optional[int]) -> torch.Tensor:
        """Compute the metric given the predictions and the targets."""
        ...


class NormalizedDCG(LightningAveragedMetric):
    """Normalized Discounted Cumulative Gain (NDCG)."""

    def _metric_fn(self, preds: torch.Tensor, target: torch.Tensor, k: Optional[int]) -> torch.Tensor:
        return torchmetrics.functional.retrieval_normalized_dcg(preds=preds, target=target, k=k)


class Recall(LightningAveragedMetric):
    """Recall metric."""

    def _metric_fn(self, preds: torch.Tensor, target: torch.Tensor, k: Optional[int]) -> torch.Tensor:
        return torchmetrics.functional.retrieval_recall(preds=preds, target=target, k=k)


class Precision(LightningAveragedMetric):
    """Precision metric."""

    def _metric_fn(self, preds: torch.Tensor, target: torch.Tensor, k: Optional[int]) -> torch.Tensor:
        return torchmetrics.functional.retrieval_precision(preds=preds, target=target, k=k)


def retrieval_metric_factory(name: str, **kwargs: Any) -> Metric:
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
        "recall": Recall,
        "precision": Precision,
    }

    cls = avail_cls[name]

    return cls(top_k=top_k, **kwargs)


class RetrievalMetricCollection(MetricCollection):
    """A collection of `torchmetrics.Metric` for information retrieval."""

    def __init__(self, metrics: Iterable[str], **kwargs: Any):
        def clean_name(x: str) -> str:
            return x.replace("@", "_")

        metrics = {clean_name(name): retrieval_metric_factory(name=name, **kwargs) for name in metrics}
        super().__init__(metrics=metrics)
