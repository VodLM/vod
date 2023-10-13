import abc
import typing as typ

import torch
from typing_extensions import Self, Type


@torch.jit.script
def _arg_first_non_zero(values: torch.Tensor) -> torch.Tensor:
    ids = torch.arange(values.shape[-1], device=values.device)
    nnz_ordered_values = torch.where(values > 0, ids, 1 + ids.max())
    return nnz_ordered_values.argmin(dim=-1)


@torch.jit.script
def _mask_rank_inputs(*, relevances: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask and sort the inputs."""
    is_mask = scores.isnan() | (scores.isinf() & (scores > 0))  # Mask NaNs and -inf
    scores = scores.masked_fill(is_mask, -torch.inf)
    relevances = relevances.masked_fill(is_mask, 0)  # Set the relevances to 0 for masked scores
    # Sort the scores and relevances, return
    sort_ids = torch.argsort(scores, dim=-1, descending=True)
    ranked_relevances = torch.gather(relevances, dim=-1, index=sort_ids)
    ranked_scores = torch.gather(scores, dim=-1, index=sort_ids)
    return ranked_relevances, ranked_scores


class _ComputeMetricFromRanked(typ.Protocol):
    """Computes a retrieval metric from relevances and scores ranked by scores."""

    def __call__(
        self,
        ranked_relevances: torch.Tensor,
        ranked_scores: torch.Tensor,
        n_positives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute."""
        ...


@torch.jit.script
def _compute_mrr(
    ranked_relevances: torch.Tensor,
    ranked_scores: torch.Tensor,  # noqa: ARG001
    n_positives: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    idx_first_non_zero = _arg_first_non_zero(ranked_relevances)
    at_least_one_target = (ranked_relevances > 0).sum(dim=-1) > 0
    mrr = 1.0 / (1 + idx_first_non_zero)
    return torch.where(at_least_one_target, mrr, 0)


@torch.jit.script
def _compute_hitrate(
    ranked_relevances: torch.Tensor,
    ranked_scores: torch.Tensor,  # noqa: ARG001
    n_positives: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    return (ranked_relevances > 0).any(dim=-1)


@torch.jit.script
def _compute_precision(
    ranked_relevances: torch.Tensor,
    ranked_scores: torch.Tensor,  # noqa: ARG001
    n_positives: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    n_retrieved_relevant = (ranked_relevances > 0).sum(dim=-1)
    n_retrieved = ranked_scores.isfinite().sum(dim=-1)
    return n_retrieved_relevant / n_retrieved


@torch.jit.script
def _compute_recall(
    ranked_relevances: torch.Tensor,
    ranked_scores: torch.Tensor,  # noqa: ARG001
    n_positives: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    n_retrieved_relevant = (ranked_relevances > 0).sum(dim=-1)
    return n_retrieved_relevant / n_positives


@torch.jit.script
def _compute_kldiv(
    ranked_relevances: torch.Tensor,
    ranked_scores: torch.Tensor,
    n_positives: torch.Tensor,
) -> torch.Tensor:
    """Compute the KL divergence between the data and the model scores."""
    is_finite = torch.isfinite(ranked_scores)
    n_positives = (ranked_relevances > 0).sum(dim=-1)
    ranked_relevances = ranked_relevances.to(ranked_scores)  # Cast
    # Compute the data log_probs, consider a unfiform distribution when no positive is present
    data_scores = ranked_relevances.masked_fill(ranked_relevances <= 0, -torch.inf)
    data_logprobs = data_scores.log_softmax(dim=-1)
    data_logprobs = torch.where((n_positives > 0).unsqueeze(-1), data_logprobs, is_finite.sum(dim=-1, keepdim=True))
    # Compute the model log_probs
    ranked_scores = ranked_scores.masked_fill(~is_finite, -torch.inf)
    model_logprobs = ranked_scores.log_softmax(dim=-1)
    # compute the KL divergence
    kl_div_terms = torch.where(
        data_logprobs.isfinite() & model_logprobs.isfinite(),
        data_logprobs.exp() * (data_logprobs - model_logprobs),
        0.0,
    )
    return kl_div_terms.sum(dim=-1)


@torch.jit.script
def _compute_ndcg(
    ranked_relevances: torch.Tensor,
    ranked_scores: torch.Tensor,
    n_positives: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    # Compute DCG (Discounted Cumulative Gain)
    log2_ranks = torch.arange(
        2,
        ranked_relevances.shape[-1] + 2,
        device=ranked_scores.device,
        dtype=ranked_scores.dtype,
    ).log2()
    dcg = torch.sum(ranked_relevances / log2_ranks, dim=-1)
    # Compute IDCG (Ideal DCG)
    sorted_relevances = torch.sort(ranked_relevances, descending=True, dim=-1).values
    idcg = torch.sum(sorted_relevances / log2_ranks, dim=-1)
    # Compute the NDCG
    return torch.where(idcg > 0, dcg / idcg, 0)


@torch.jit.script
def prepare_for_metric_computation(
    *,
    relevances: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare inputs for retrieval metric computation."""
    n_positives = (relevances > 0).sum(dim=-1)
    ranked_relevances, ranked_scores = _mask_rank_inputs(relevances=relevances, scores=scores)
    if topk > 0:
        ranked_relevances = ranked_relevances[..., :topk]
        ranked_scores = ranked_scores[..., :topk]

    return ranked_relevances, ranked_scores, n_positives


class ComputeMetric(abc.ABC):
    """Computes retrieval metrics."""

    _compute_from_ranked: _ComputeMetricFromRanked

    @classmethod
    def __call__(
        cls: Type[Self], *, relevances: torch.Tensor, scores: torch.Tensor, topk: None | int = None
    ) -> torch.Tensor:
        """Compute a retrieval metric."""
        ranked_relevances, ranked_scores, n_positives = prepare_for_metric_computation(
            relevances=relevances,
            scores=scores,
            topk=topk or -1,
        )
        return cls._compute_from_ranked(
            ranked_relevances=ranked_relevances,
            ranked_scores=ranked_scores,
            n_positives=n_positives,
        )


class compute_mrr(ComputeMetric):  # noqa: N801
    """Computes the Mean Reciprocal Rank (MRR)."""

    _compute_from_ranked = staticmethod(_compute_mrr)


class compute_hitrate(ComputeMetric):  # noqa: N801
    """Computes the Hit Rate (HR)."""

    _compute_from_ranked = staticmethod(_compute_hitrate)


class compute_precision(ComputeMetric):  # noqa: N801
    """Computes the Precision."""

    _compute_from_ranked = staticmethod(_compute_precision)


class compute_recall(ComputeMetric):  # noqa: N801
    """Computes the Recall."""

    _compute_from_ranked = staticmethod(_compute_recall)


class compute_ndcg(ComputeMetric):  # noqa: N801
    """Computes the Normalized Discounted Cumulative Gain (NDCG)."""

    _compute_from_ranked = staticmethod(_compute_ndcg)


class compute_kldiv(ComputeMetric):  # noqa: N801
    """Computes the Kullback-Leibler divergence (KLD)."""

    _compute_from_ranked = staticmethod(_compute_kldiv)
