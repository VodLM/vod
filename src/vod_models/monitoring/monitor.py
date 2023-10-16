import abc
import typing as typ

import torch
import vod_types as vt

from .aggregator import Agregator, MeanAggregator
from .functional import (
    _compute_entropy,
    _compute_hitrate,
    _compute_kldiv,
    _compute_max,
    _compute_min,
    _compute_mrr,
    _compute_ndcg,
    _compute_precision,
    _compute_recall,
    _ComputeMetricFromRanked,
    prepare_for_metric_computation,
)

RETRIEVAL_METRICS = {
    "mrr": _compute_mrr,
    "ndcg": _compute_ndcg,
    "hitrate": _compute_hitrate,
    "recall": _compute_recall,
    "precision": _compute_precision,
    "kldiv": _compute_kldiv,
    "min": _compute_min,
    "max": _compute_max,
    "entropy": _compute_entropy,
}


class Monitor(abc.ABC, torch.nn.Module):
    """Monitor retrieval performances."""

    aggregators: typ.Mapping[str, Agregator]

    @abc.abstractmethod
    def update(self, batch: typ.Mapping[str, typ.Any], model_output: typ.Mapping[str, typ.Any]) -> None:
        """Compute metrics and update the aggregators."""
        ...

    def synchronize(self) -> None:
        """Synchronize aggregators between process."""
        if torch.distributed.is_initialized():
            for agg in self.aggregators.values():
                agg.all_reduce()

    def reset(self) -> None:
        """Reset aggregators."""
        for agg in self.aggregators.values():
            agg.reset()

    def get(self) -> dict[str, torch.Tensor]:
        """Get values from all aggregators."""
        return {name: agg.get() for name, agg in self.aggregators.items()}

    def compute(self, synchronize: bool = True) -> dict[str, torch.Tensor]:
        """Sync, get values and reset."""
        if synchronize:
            self.synchronize()
        outputs: dict[str, torch.Tensor] = {}
        for name, agg in self.aggregators.items():
            outputs[name] = agg.get()
            agg.reset()

        return outputs


class RetrievalMonitor(Monitor):
    """Monitor retrieval performances."""

    ops: dict[str, tuple[_ComputeMetricFromRanked, None | int]]

    def __init__(self, metrics: list[str]) -> None:
        super().__init__()
        self.ops = {m: _parse_metric_name(m) for m in metrics}
        self.aggregators = torch.nn.ModuleDict({metric: MeanAggregator() for metric in self.ops})  # type: ignore

    @torch.no_grad()
    def update(
        self,
        batch: vt.RealmBatch | typ.Mapping[str, typ.Any],
        model_output: vt.ModelOutput | typ.Mapping[str, typ.Any],
    ) -> None:
        """Compute metrics and update the aggregators."""
        batch = vt.RealmBatch.cast(batch)
        model_output = vt.ModelOutput.cast(model_output)

        # Rank the relevances and scores by decreasing score value
        ranked_relevances, ranked_scores, n_positives = prepare_for_metric_computation(
            relevances=batch.section__label, scores=model_output.retriever_scores, topk=-1
        )

        # Compute each metric
        for key, (op, topk) in self.ops.items():
            values = op(
                ranked_relevances=ranked_relevances[..., :topk],
                ranked_scores=ranked_scores[..., :topk],
                n_positives=n_positives,
            )
            self.aggregators[key].update(values)


def _parse_metric_name(name: str) -> tuple[_ComputeMetricFromRanked, None | int]:
    """Instantiate a torchmetrics retrieval metric from a string name."""
    if "_" in name:
        *parts, k = name.split("_")
        top_k = int(k)
        name = "_".join(parts)
    else:
        top_k = None

    return RETRIEVAL_METRICS[name], top_k
