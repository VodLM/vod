import dataclasses

import numba
import numpy as np
import numpy.typing as npt
import vod_types as vt
from vod_dataloaders.core import numpy_ops as npo


@dataclasses.dataclass(frozen=True)
class PrioritySampledSections:
    """A holder for the samples and the log-weights."""

    batch: vt.RetrievalBatch
    log_weights: np.ndarray
    max_sampling_id: np.ndarray
    lse_pos: np.ndarray
    lse_neg: np.ndarray
    raw_scores: dict[str, np.ndarray]


def sample_search_results(
    *,
    search_results: vt.RetrievalBatch,
    raw_scores: dict[str, np.ndarray],
    total: None | int,
    max_pos_sections: None | int,
    temperature: float = 1.0,
    max_support_size: None | int = None,
) -> PrioritySampledSections:
    """Sample the positive and negative sections using per-label priority sampling.

    See `labeled_priority_sampling` for more details.
    """
    total = total or search_results.shape[-1]
    max_pos_sections = max_pos_sections or total

    # Gather reference attributes (inputs)
    indices_ref: np.ndarray = search_results.indices
    scores_ref: np.ndarray = search_results.scores
    if search_results.labels is None:
        labels_ref: np.ndarray = np.zeros_like(search_results.scores, dtype=np.bool_)
    else:
        labels_ref: np.ndarray = search_results.labels > 0  # <- make sure to cast to bool

    # Sample section using priority sampling
    local_ids, log_weights, labels, constants = labeled_priority_sampling(
        scores=scores_ref,
        labels=labels_ref,
        k_positive=max_pos_sections,
        k_total=total,
        normalized=True,  # <- self-normalized the importance weights
        temperature=temperature,
        max_support_size=max_support_size,
    )

    # Fetch the sampled `indices` and `scores` values associated to the sampled `local_ids`
    indices = np.take_along_axis(indices_ref, local_ids, axis=-1)
    scores = np.take_along_axis(scores_ref, local_ids, axis=-1)

    # Fetch the `raw_scores`
    sampled_raw_scores = {}
    for key, scores_key in raw_scores.items():
        sampled_raw_scores[key] = np.take_along_axis(scores_key, local_ids, axis=-1)

    # Compute the rank of the document sampled with the smaller score.
    # This is used to debug the sampling step.
    min_score = np.amin(np.where(np.isfinite(scores), scores, np.inf), axis=-1)
    max_sampling_id = np.sum((scores_ref >= min_score[..., None]).astype(np.float32), axis=-1)

    return PrioritySampledSections(
        batch=vt.RetrievalBatch(
            indices=indices,
            scores=scores,
            labels=labels,
        ),
        max_sampling_id=max_sampling_id,
        lse_pos=constants[..., 0],
        lse_neg=constants[..., 1],
        log_weights=log_weights,
        raw_scores=sampled_raw_scores,
    )


def labeled_priority_sampling(
    scores: npt.NDArray[npo.Float],
    labels: npt.NDArray[npo.Bool],
    k_positive: int = 1,
    k_total: int = 2,
    normalized: bool = True,
    temperature: float = 1.0,
    max_support_size: None | int = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[npo.Float], npt.NDArray[npo.Bool], npt.NDArray[npo.Float]]:
    """Sample search results using Priority Sampling for each label value {0, 1}.

    Priority sampling allows sampling from a Categorical distribution without replacement:
        z_1, ..., z_k ~ Categorical(p_1, ..., p_n)

    The samples are drawn with corresponding weights and are normalized (sum to 1):
        s_1, ..., s_k <- z_1, ..., z_k

    Using the weights, we can estimate an unbiased estimate of an expectation F = E_p[f(z)] using
        F = sum_i s_i f(z_i)               # <- unbiased estimate of E_p[f(z)] if temperature=1.0 and normalized=False

    Args:
        scores: The unformalized log p(z) scores.
        labels: The labels of the samples.
        k_positive: The number of positive samples to draw.
                    warning: might be increased if there are not enough negative samples.
        k_total: The total number of samples to draw.
        normalized: return the normalized weights (sum to 1) or the unnormalized weights. Default: True
        temperature: The temperature of the softmax, zero means deterministic (top-K). Default: 1.0
        max_support_size: The maximum number of samples to consider (truncated sampling). Default: None (no truncation)

    Returns:
        sample indices, log weights, labels, log-Normalizing constant

    NOTE: Labels: a sample is `positive` if its label is `>0`, `negative` otherwise.
          we run priority sampling for each sets {i | l_i = 0} and {i | l_i = 1}
    NOTE: Normalization: if `normalize==True`, the weights are normalized to sum to 1, for each label.
          This is an instance of self-normalized importance sampling. This leads a slight bias in the estimate
          of the expectation, but it is negligible in practice.
    NOTE: References:
          Original paper: https://arxiv.org/abs/cs/0509026
          Tim Vieira's review: https://timvieira.github.io/blog/post/2017/07/03/estimating-means-in-a-finite-universe/
    NOTE: Truncation: if `max_support_size` is specified, the distribution is truncated to the top `max_support_size`.
          This allows restricting the sampling to a subset of the distribution and avoid sampling from the tail.
          This controls an exploration-exploitation trade-off, as discussed in https://arxiv.org/abs/2210.06345
    NOTE: Temperature: using a temperature != 1.0 will yield to unbiased estimates of the expectation!
    """
    max_support_size = max_support_size or -1
    if max_support_size >= 0:
        max_support_size = max(max_support_size, k_total)
    if scores.ndim == 1:
        return labeled_priority_sampling_1d(
            scores,
            labels,
            k_positive,
            k_total,
            normalized=normalized,
            temperature=temperature,
            max_support_size=max_support_size,
        )
    elif scores.ndim == 2:  # noqa
        return labeled_priority_sampling_2d(
            scores,
            labels,
            k_positive,
            k_total,
            normalized=normalized,
            temperature=temperature,
            max_support_size=max_support_size,
        )
    else:
        raise ValueError(f"Expected a 1D or 2D array. Got {scores.ndim}D.")


@numba.njit(fastmath=True, cache=npo.CACHE_NUMBA_JIT, nogil=True)
def _priority_sampling_1d(
    scores: npt.NDArray[npo.Float],
    exponential_noise_: npt.NDArray[npo.Float],
    k: int = 1,
    temperature: float = 1.0,
    max_support_size: int = -1,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[npo.Float], npo.Float]:
    """Truncated priority sampling."""
    _dtype = scores.dtype
    temperature_inv = _dtype.type(temperature if temperature > 0 else 1.0)

    #### Start - Compute the log-probabilities
    log_p: npt.NDArray[npo.Float] = scores.copy()
    npo.mul_1d_(log_p, temperature_inv)
    # Truncate the distribution to the top `max_support_size`
    if (max_support_size > 0) and (len(log_p) > max_support_size):
        threshold = np.sort(log_p)[-max_support_size]
        npo.masked_fill_1d_(log_p >= threshold, log_p, _dtype.type(-np.inf))
    # Normalize the scores
    npo.log_softmax_1d_(log_p)
    #### End - Compute the log-probabilities

    # Compute the normalization constant `LogSum(p(z))``
    log_norm_const = np.log(np.sum(np.exp(log_p)))

    # Log exponential noise, u ~ Exp(1)
    log_u: npt.NDArray[npo.Float] = np.log(exponential_noise_)

    # Compute the log keys
    if temperature > 0:
        log_keys: npt.NDArray[npo.Float] = log_p - log_u  # type: ignore
    else:
        log_keys: npt.NDArray[npo.Float] = log_p.copy()

    # Take the top-(K+1) samples sorted by largest keys
    sorted_ids: npt.NDArray[np.int64] = np.argsort(-log_keys)[: k + 1]

    # Find the threshold (-inf if not enough samples)
    if k < log_p.shape[-1]:
        tau_idx: np.int64 = sorted_ids[-1]
        log_tau: npo.Float = log_keys[tau_idx]
    else:
        log_tau: npo.Float = _dtype.type(-np.inf)

    # Take the top-K samples
    sorted_ids = sorted_ids[:k]
    log_pi: npt.NDArray[npo.Float] = np.take(log_p, sorted_ids)

    # Compute the log importance weights
    if log_tau > -np.inf:
        log_qz = log_pi.copy()
        npo.add_1d_(log_qz, -log_tau)
        log_qz = np.log1p(-np.exp(-np.exp(log_qz)))
        log_weights: npt.NDArray[npo.Float] = log_pi - log_qz  # type: ignore
    else:
        log_weights: npt.NDArray[npo.Float] = log_pi.copy()

    # return the samples, the log weights and the LogSumExp of the logprobs (normalazing constant)
    return sorted_ids, log_weights, log_norm_const  # type: ignore


def priority_sampling_1d(
    scores: npt.NDArray[npo.Float],
    k: int = 1,
    temperature: float = 1.0,
    max_support_size: int = -1,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[npo.Float]]:
    """Sample from unformalized log p(z) using priority sampling."""
    max_support_size = max_support_size or -1
    if scores.ndim > 1:  # type: ignore
        raise ValueError("Expected a 1D array.")

    # sample the noise and compute priority sampling
    noise = np.random.exponential(size=scores.shape).astype(scores.dtype)
    sorted_ids, log_weights, _ = _priority_sampling_1d(
        scores,
        noise,
        k,
        temperature=temperature,
        max_support_size=max_support_size,
    )
    return sorted_ids, log_weights


@numba.njit(fastmath=True, cache=npo.CACHE_NUMBA_JIT, nogil=True)
def _labeled_priority_sampling_1d_(  # noqa: PLR0913
    scores: npt.NDArray[npo.Float],
    labels: npt.NDArray[npo.Bool],
    exponential_noise_: npt.NDArray[npo.Float],
    k_positive: int,
    k_total: int,
    out_samples_: npt.NDArray[npo.Int],
    out_log_weights_: npt.NDArray[npo.Float],
    out_labels_: npt.NDArray[npo.Bool],
    out_lse_: npt.NDArray[npo.Float],
    normalized: bool,
    temperature: float = 1.0,
    max_support_size: int = -1,
) -> None:
    indices = np.arange(len(scores))
    labels = labels > 0  # type: ignore - make sure to cast to bool
    labels_ = ~labels
    is_inf = np.isinf(scores)
    n_pos_finite = np.sum(labels & ~is_inf)
    n_neg_finite = np.sum(labels_ & ~is_inf)

    # decrease k_total to match the number of available samples
    # note that we write to a buffer of size k_total, so the ouput shape is not affected
    k_total = len(scores) if k_total > len(scores) else k_total

    # if there are not enough positive samples, decrease the number of positive samples
    if n_pos_finite < k_positive:
        n_pos_finite = k_positive

    # if there are not enough negative samples, increase the number of positive samples
    # so we can reach the desired number of total samples
    if n_neg_finite < k_total - k_positive:
        k_positive = k_total - n_neg_finite  # type: ignore

    # Priority samples the positive labels
    pos_samples_, pos_log_weights, pos_lse = _priority_sampling_1d(
        scores[labels],
        exponential_noise_[labels],
        k_positive,
        temperature=temperature,
        max_support_size=max_support_size,
    )
    pos_samples = indices[labels][pos_samples_]
    if normalized and len(pos_samples) > 0:
        npo.log_softmax_1d_(pos_log_weights)

    # Priority samples the negative labels
    neg_samples_, neg_log_weights, neg_lse = _priority_sampling_1d(
        scores[labels_],
        exponential_noise_[labels_],
        k_total - len(pos_samples_),
        temperature=temperature,
        max_support_size=max_support_size,
    )
    neg_samples = indices[labels_][neg_samples_]
    if normalized and len(neg_samples) > 0:
        npo.log_softmax_1d_(neg_log_weights)

    # Fills the LogSumExp of the logprobs (normalazing constant)
    assert out_lse_.shape[0] == 2  # noqa: S101, PLR2004
    out_lse_[0] = pos_lse
    out_lse_[1] = neg_lse

    # return the samples and the log weights
    j: int = 0
    for i in range(len(pos_samples)):
        out_samples_[j] = pos_samples[i]
        out_log_weights_[j] = pos_log_weights[i]
        out_labels_[j] = 1
        j += 1
    for i in range(len(neg_samples)):
        out_samples_[j] = neg_samples[i]
        out_log_weights_[j] = neg_log_weights[i]
        out_labels_[j] = 0
        j += 1


@numba.njit(parallel=True, fastmath=True, cache=npo.CACHE_NUMBA_JIT, nogil=True)
def _labeled_priority_sampling_2d_(  # noqa: PLR0913
    scores: npt.NDArray[npo.Float],
    labels: npt.NDArray[npo.Bool],
    exponential_noise_: npt.NDArray[npo.Float],
    k_positive: int,
    k_total: int,
    out_samples_: npt.NDArray[npo.Int],
    out_log_weights_: npt.NDArray[npo.Float],
    out_labels_: npt.NDArray[npo.Bool],
    out_lse_: npt.NDArray[npo.Float],
    normalized: bool,
    temperature: float = 1.0,
    max_support_size: int = -1,
) -> None:
    for i in numba.prange(len(scores)):
        _labeled_priority_sampling_1d_(
            scores[i],
            labels[i],
            exponential_noise_[i],
            k_positive,
            k_total,
            out_samples_=out_samples_[i],
            out_log_weights_=out_log_weights_[i],
            out_labels_=out_labels_[i],
            out_lse_=out_lse_[i],
            normalized=normalized,
            temperature=temperature,
            max_support_size=max_support_size,
        )


def labeled_priority_sampling_1d(
    scores: npt.NDArray[npo.Float],
    labels: npt.NDArray[npo.Bool],
    k_positive: int = 1,
    k_total: int = 2,
    normalized: bool = True,
    temperature: float = 1.0,
    max_support_size: None | int = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[npo.Float], npt.NDArray[npo.Bool], npt.NDArray[npo.Float]]:
    """Sample from unformalized log p(z) using priority sampling."""
    noise_ = np.random.exponential(size=scores.shape).astype(scores.dtype)
    samples_ = np.full(k_total, -1, dtype=np.int64)
    log_weights_ = np.full(k_total, -np.inf, dtype=scores.dtype)
    labels_: npt.NDArray[npo.Bool] = np.full(k_total, 0, dtype=np.bool_)  # type: ignore
    lse_: npt.NDArray[npo.Float] = np.full((2,), 0, dtype=scores.dtype)
    max_support_size = max_support_size or -1
    _labeled_priority_sampling_1d_(
        scores,
        labels,
        noise_,
        k_positive,
        k_total,
        out_samples_=samples_,
        out_log_weights_=log_weights_,
        out_labels_=labels_,
        out_lse_=lse_,
        normalized=normalized,
        temperature=temperature,
        max_support_size=max_support_size,
    )
    return samples_, log_weights_, labels_, lse_


def labeled_priority_sampling_2d(
    scores: npt.NDArray[npo.Float],
    labels: npt.NDArray[npo.Bool],
    k_positive: int = 1,
    k_total: int = 2,
    normalized: bool = True,
    temperature: float = 1.0,
    max_support_size: None | int = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[npo.Float], npt.NDArray[npo.Bool], npt.NDArray[npo.Float]]:
    """Sample from unformalized log p(z) using priority sampling."""
    noise_ = np.random.exponential(size=scores.shape).astype(scores.dtype)
    samples_ = np.full((len(scores), k_total), -1, dtype=np.int64)
    log_weights_ = np.full((len(scores), k_total), -np.inf, dtype=scores.dtype)
    labels_: npt.NDArray[npo.Bool] = np.full((len(scores), k_total), 0, dtype=np.bool_)  # type: ignore
    lse_: npt.NDArray[npo.Float] = np.full((len(scores), 2), 0, dtype=scores.dtype)
    max_support_size = max_support_size or -1
    _labeled_priority_sampling_2d_(
        scores,
        labels,
        noise_,
        k_positive,
        k_total,
        out_samples_=samples_,
        out_log_weights_=log_weights_,
        out_labels_=labels_,
        out_lse_=lse_,
        normalized=normalized,
        temperature=temperature,
        max_support_size=max_support_size,
    )
    return samples_, log_weights_, labels_, lse_
