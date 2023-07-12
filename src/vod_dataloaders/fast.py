from typing import Optional

import numba
import numpy as np
from numpy import typing as npt
from raffle_ds_research.tools import index_tools
from typing_extensions import TypeVar

DtypeVar = TypeVar("DtypeVar", bound=np.dtype)

CACHE_NUMBA_JIT = True


def _default_fill_value(dt: np.dtype) -> float | int:
    """Return the default fill value for a given dtype."""
    if dt.kind == "f":
        return np.nan

    return -1


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT)
def _nopy_gather_values_1d(
    queries: npt.NDArray[np.int_],
    keys: npt.NDArray[np.int_],
    values: npt.NDArray,
    output: npt.NDArray,
) -> None:
    """Gather values from an array of keys."""
    for i in numba.prange(len(queries)):
        query = queries[i]
        for j in range(len(keys)):
            key = keys[j]
            if key == query:
                output[i] = values[j]
                break


def gather_values_1d(
    queries: npt.NDArray[np.int_],
    keys: npt.NDArray[np.int_],
    values: npt.NDArray,
    fill_value: Optional[float | int] = None,
) -> npt.NDArray:
    """Gather values from an array indexed by keys."""
    if fill_value is None:
        fill_value = _default_fill_value(values.dtype)
    output = np.full_like(queries, fill_value=fill_value, dtype=values.dtype)
    queries.setflags(write=False)
    keys.setflags(write=False)
    values.setflags(write=False)
    _nopy_gather_values_1d(queries, keys, values, output)
    return output


@numba.njit(parallel=True, fastmath=True, cache=CACHE_NUMBA_JIT)
def _nopy_gather_values_2d(
    queries: npt.NDArray[np.int_],
    keys: npt.NDArray[np.int_],
    values: npt.NDArray,
    output: npt.NDArray,
) -> None:
    bs = len(queries)
    if len(keys) != bs:
        raise ValueError(f"Expected keys to have length {bs}. Found: {len(keys)}")
    if len(values) != bs:
        raise ValueError(f"Expected values to have length {bs}. Found: {len(values)}")

    for i in numba.prange(bs):
        _nopy_gather_values_1d(queries[i], keys[i], values[i], output[i])


@numba.njit(parallel=True, fastmath=True)
def _nopy_gather_values_2d_from_1d(
    queries: npt.NDArray[np.int_],
    keys: npt.NDArray[np.int_],
    values: npt.NDArray,
    output: npt.NDArray,
) -> None:
    bs = len(queries)
    if keys.ndim != 1:
        raise ValueError(f"Expected keys to have ndim 1. Found: {keys.ndim}")
    if values.ndim != 1:
        raise ValueError(f"Expected values to have ndim 1. Found: {values.ndim}")

    for i in numba.prange(bs):
        _nopy_gather_values_1d(queries[i], keys, values, output[i])


def gather_values_2d(
    queries: npt.NDArray[np.int_],
    keys: npt.NDArray[np.int_],
    values: npt.NDArray,
    fill_value: Optional[float | int] = None,
) -> npt.NDArray:
    """Gather values from an array indexed by keys."""
    if fill_value is None:
        fill_value = _default_fill_value(values.dtype)
    output = np.full_like(queries, fill_value=fill_value, dtype=values.dtype)
    queries.setflags(write=False)
    keys.setflags(write=False)
    values.setflags(write=False)
    _nopy_gather_values_2d(queries, keys, values, output)
    return output


def gather_values_2d_from_1d(
    queries: npt.NDArray[np.int_],
    keys: npt.NDArray[np.int_],
    values: npt.NDArray,
    fill_value: Optional[float | int] = None,
) -> npt.NDArray:
    """Gather values from an array indexed by keys."""
    if fill_value is None:
        fill_value = _default_fill_value(values.dtype)
    output = np.full_like(queries, fill_value=fill_value, dtype=values.dtype)
    queries.setflags(write=False)
    keys.setflags(write=False)
    values.setflags(write=False)
    _nopy_gather_values_2d_from_1d(queries, keys, values, output)
    return output


def gather_values_by_indices(
    queries: npt.NDArray[np.int_],
    indices: npt.NDArray[np.int_],
    values: npt.NDArray,
    fill_value: Optional[float | int] = None,
) -> npt.NDArray:
    """Gather values from an array indexed by keys."""
    if queries.ndim == 1:
        return gather_values_1d(queries, indices, values, fill_value=fill_value)
    if queries.ndim == 2:  # noqa: PLR2004
        if indices.ndim == 1:
            return gather_values_2d_from_1d(queries, indices, values, fill_value=fill_value)
        if indices.ndim == 2:  # noqa: PLR2004
            return gather_values_2d(queries, indices, values, fill_value=fill_value)

        raise ValueError(f"Expected indices to have ndim 1 or 2. Found: {indices.ndim}")

    raise ValueError(f"Expected queries to have ndim 1 or 2. Found: {queries.ndim}")


def merge_search_results(
    search_results: dict[str, index_tools.RetrievalBatch[npt.NDArray]],
    weights: Optional[dict[str, float]] = None,
) -> tuple[index_tools.RetrievalBatch, dict[str, npt.NDArray]]:
    """Merge search results with weights."""
    if weights is None:
        weights = {k: 1.0 for k in search_results}
    elif not set(weights) >= set(search_results):
        raise ValueError(f"Expected weights to have keys {set(search_results)}. Found: {set(weights)}")

    if len(search_results) == 1:
        key = list(search_results.keys())[0]
        result = search_results[key]
        weighted_result = result * weights[key]
        return weighted_result, {key: search_results[key].scores}

    ulengths = {len(v.scores) for v in search_results.values()}
    if len(ulengths) != 1:
        raise ValueError(f"All scores must have the same length. Found: {ulengths}")

    return _merge_n_search_results(search_results, weights)


def normalize_scores_(
    search_results: dict[str, index_tools.RetrievalBatch[npt.NDArray]],
    offset: float = 0.0,
) -> None:
    """Subtract the minimum score from all score to allow consistent aggregation."""
    for key, result in search_results.items():
        search_results[key].scores = _subtract_min_score(result.scores, offset=offset)


def _subtract_min_score(scores: npt.NDArray, offset: float) -> npt.NDArray:
    non_nan_scores = np.where(np.isinf(scores) | np.isnan(scores), np.inf, scores)
    min_score = np.amin(non_nan_scores, axis=1, keepdims=True)
    return scores - min_score + offset


def _merge_n_search_results(
    search_results: dict[str, index_tools.RetrievalBatch[npt.NDArray]],
    weights: dict[str, float],
) -> tuple[index_tools.RetrievalBatch, dict[str, npt.NDArray]]:
    keys = list(search_results.keys())
    first_result = search_results[keys[0]]
    first_weight = weights[keys[0]]
    output = first_result * first_weight
    for j in range(1, len(keys)):
        other = search_results[keys[j]]
        weight = weights[keys[j]]
        output = _merge_two_search_results(output, other * weight)

    # gather scores
    raw_scores = {}
    for key, value in search_results.items():
        raw_scores[key] = gather_values_by_indices(queries=output.indices, indices=value.indices, values=value.scores)

    # lookup labels
    output.labels = None
    for r in search_results.values():
        if r.labels is not None:
            output.labels = gather_values_by_indices(
                queries=output.indices,
                indices=r.indices,
                values=r.labels,
                fill_value=-1,
            )

    return output, raw_scores


def _merge_two_search_results(
    a: index_tools.RetrievalBatch[npt.NDArray],
    b: index_tools.RetrievalBatch[npt.NDArray],
) -> index_tools.RetrievalBatch[npt.NDArray]:
    """Merge two search results."""
    scores, indices = _nopy_merge_two_search_results(a.scores, a.indices, b.scores, b.indices)
    return index_tools.RetrievalBatch(scores=scores, indices=indices)


@numba.njit(fastmath=True)
def _search_1d_arr(arr: npt.NDArray, x: int | float) -> int:
    """Search for values in a sorted array."""
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")

    for i, v in enumerate(arr):
        if v == x:
            return i

    return -1


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT)
def _write_1d_arr(
    indices: npt.NDArray,
    scores: npt.NDArray,
    index: int,
    score: int | float,
    cursor: int,
) -> int:
    """Write a value to an array at a given index."""
    if index < 0:
        return cursor

    # check if the value is already present
    found_index = _search_1d_arr(indices, index)
    if found_index < 0:
        scores[cursor] = score
        indices[cursor] = index
        cursor = cursor + 1
    else:
        scores[found_index] = score + scores[found_index]

    return cursor


@numba.njit(parallel=True, fastmath=True, cache=CACHE_NUMBA_JIT)
def _nopy_merge_two_search_results(
    a_scores: npt.NDArray,
    a_indices: npt.NDArray,
    b_scores: npt.NDArray,
    b_indices: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Merge two search results."""
    scores = np.full(
        (a_scores.shape[0], (a_scores.shape[1] + b_scores.shape[1])),
        fill_value=np.nan,
        dtype=a_scores.dtype,
    )
    indices = np.full(
        (a_indices.shape[0], (a_indices.shape[1] + b_indices.shape[1])),
        fill_value=-1,
        dtype=a_indices.dtype,
    )

    # fill the output arrays
    cursors = np.zeros(a_scores.shape[0], dtype=np.int64)
    for i in numba.prange(a_scores.shape[0]):
        scores_i = scores[i]
        indices_i = indices[i]
        a_scores_i = a_scores[i]
        a_indices_i = a_indices[i]
        b_scores_i = b_scores[i]
        b_indices_i = b_indices[i]
        cursor = 0

        # write data from `a`
        for j in numba.prange(a_scores_i.shape[0]):
            cursor = _write_1d_arr(
                indices_i,
                scores_i,
                a_indices_i[j],
                a_scores_i[j],
                cursor,
            )

        # write data from `b`
        for j in numba.prange(b_scores_i.shape[0]):
            cursor = _write_1d_arr(
                indices_i,
                scores_i,
                b_indices_i[j],
                b_scores_i[j],
                cursor,
            )
        cursors[i] = cursor

    # truncate the output arrays
    max_cursor = np.max(cursors)
    scores = scores[:, : max_cursor + 1]
    indices = indices[:, : max_cursor + 1]

    return scores, indices


def log_softmax(x: npt.NDArray, dim: int = -1) -> npt.NDArray:
    """Compute log-softmax values for a given tensor."""
    x_safe = np.where(np.isnan(x), -np.inf, x)

    # substract the max
    x_max = np.max(x_safe, axis=dim, keepdims=True)
    x_max[np.isinf(x_max)] = 0
    x_safe = x_safe - x_max

    # compute the log sum exp
    lse = np.log(np.sum(np.exp(x_safe), axis=dim, keepdims=True))

    # compute the log softmax
    return x_safe - lse


@numba.njit(fastmath=True)
def _log_softmax_1d(x: npt.NDArray) -> npt.NDArray:
    """Compute log-softmax values for a given tensor."""
    if x.ndim != 1:
        raise ValueError("Expected a 1D array.")
    x_safe = np.where(np.isnan(x), -np.inf, x)

    # substract the max
    x_max = np.max(x_safe)
    x_max = 0 if np.isinf(x_max) else x_max
    x_safe = x_safe - x_max

    # compute the log sum exp
    lse = np.log(np.sum(np.exp(x_safe)))

    # compute the log softmax
    return x_safe - lse


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT)
def _gumbel_1d_like(x: npt.NDArray, eps: float = 1e-18) -> npt.NDArray:
    """Sample Gumbel(0, 1) random variables."""
    if x.ndim != 1:
        raise ValueError("Expected a 1D array.")

    noise = np.random.uniform(low=eps, high=1.0 - eps, size=x.shape)
    return -np.log(-np.log(noise))


def sample(
    search_results: index_tools.RetrievalBatch[npt.NDArray],
    total: int,
    n_positives: Optional[int] = None,
    temperature: float = 0.0,
    max_support_size: Optional[int] = None,
) -> index_tools.RetrievalBatch[npt.NDArray]:
    """Sample search results."""
    bs = search_results.scores.shape[0]
    total = total or search_results.scores.shape[1]
    n_positives = n_positives or total
    total = min(total, search_results.scores.shape[1])
    if search_results.labels is None:
        is_positive = np.zeros_like(search_results.indices, dtype=np.bool)
    else:
        is_positive = search_results.labels > 0

    # Set read-only flags
    search_results.scores.setflags(write=False)
    search_results.indices.setflags(write=False)
    is_positive.setflags(write=False)

    # Instantiate the output buffer
    output = index_tools.RetrievalBatch(
        scores=np.full((bs, total), fill_value=np.nan),
        indices=np.full((bs, total), fill_value=-1),
        labels=np.full((bs, total), fill_value=-1),
    )

    _sample_2d(
        scores=search_results.scores,
        indices=search_results.indices,
        is_positive=is_positive,
        output_scores=output.scores,
        output_indices=output.indices,
        output_labels=output.labels,  # type: ignore
        n_positives=n_positives,
        temperature=temperature,
        max_support_size=max_support_size or -1,
    )

    return output


@numba.njit(parallel=True, fastmath=True, cache=CACHE_NUMBA_JIT)
def _sample_2d(  # noqa: PLR0913
    scores: npt.NDArray,
    indices: npt.NDArray,
    is_positive: npt.NDArray,
    output_scores: npt.NDArray,
    output_indices: npt.NDArray,
    output_labels: npt.NDArray,
    n_positives: int,
    temperature: float,
    max_support_size: int,
) -> None:
    for i in numba.prange(scores.shape[0]):
        _sample_1d(
            scores=scores[i],
            indices=indices[i],
            is_positive=is_positive[i],
            output_scores=output_scores[i],
            output_indices=output_indices[i],
            output_labels=output_labels[i],
            n_positives=n_positives,
            temperature=temperature,
            max_support_size=max_support_size,
        )


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT)
def _sample_1d(  # noqa: PLR0913, C901
    scores: npt.NDArray,
    indices: npt.NDArray,
    is_positive: npt.NDArray,
    output_scores: npt.NDArray,
    output_indices: npt.NDArray,
    output_labels: npt.NDArray,
    n_positives: int,
    temperature: float,
    max_support_size: int = -1,
) -> None:
    """Sample indices based on scores.

    This function uses the Gumbel-Max trick to sample from the corresponding distributions.
    Gumbel-Max: https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/.
    """
    positive_scores = np.where(is_positive, scores, -np.inf)
    negative_scores = np.where(~is_positive, scores, -np.inf)

    # Truncate the distribution
    if max_support_size > 0:
        positive_cutoff = np.sort(positive_scores)[-max_support_size]
        negative_cutoff = np.sort(negative_scores)[-max_support_size]
        positive_scores = np.where(positive_scores >= positive_cutoff, positive_scores, -np.inf)
        negative_scores = np.where(negative_scores >= negative_cutoff, negative_scores, -np.inf)

    # Add perturbation (Sampling)
    if temperature > 0:
        positive_scores = _log_softmax_1d(positive_scores / temperature)
        positive_scores += _gumbel_1d_like(positive_scores)
        negative_scores = _log_softmax_1d(negative_scores / temperature)
        negative_scores += _gumbel_1d_like(negative_scores)

    cursor = 0
    # Sample positives
    if np.sum(is_positive) > 0:
        sort_ids = np.argsort(positive_scores)[::-1]
        for i in sort_ids:
            if cursor >= n_positives:
                break
            scores_i = scores[i]
            if scores_i < 0 and np.isinf(scores_i):
                break  # scores are sorted, so we can break here
            indices_i = indices[i]
            is_positive_i = is_positive[i]
            if indices_i < 0 or not is_positive_i:
                continue

            output_scores[cursor] = scores_i
            output_indices[cursor] = indices_i
            output_labels[cursor] = 1
            cursor += 1

    # Sample the remaining negatives
    sort_ids = np.argsort(negative_scores)[::-1]
    for i in sort_ids:
        if cursor >= output_scores.shape[0]:
            break
        scores_i = scores[i]
        indices_i = indices[i]
        is_positive_i = is_positive[i]
        if indices_i < 0 or is_positive_i:
            continue

        output_scores[cursor] = scores_i
        output_indices[cursor] = indices_i
        output_labels[cursor] = 0
        cursor += 1


# if __name__ == "__main__":
#     import rich

#     bs = 1
#     dim = 10
#     data = {
#         "a": index_tools.RetrievalBatch(
#             scores=10 + np.random.randn(bs, dim).astype(np.float32),
#             indices=np.random.randint(-1, dim, size=(bs, dim)),
#             labels=np.random.randint(-1, 2, size=(bs, dim)),
#         ),
#         "b": index_tools.RetrievalBatch(
#             scores=10 + np.random.randn(bs, dim + 1).astype(np.float32),
#             indices=np.random.randint(-1, dim, size=(bs, dim + 1)),
#         ),
#         "c": index_tools.RetrievalBatch(
#             scores=10 + np.random.randn(bs, dim + 2).astype(np.float32),
#             indices=np.random.randint(-1, dim, size=(bs, dim + 2)),
#         ),
#     }
#     rich.print(data)

#     output, raw = merge_search_results(data, weights={"a": 100, "b": 1.0, "c": -100})

#     rich.print("=== OUTPUT ===")
#     rich.print(output)
#     rich.print(raw)
#     rich.print({k: log_softmax(v) for k, v in raw.items()})

#     rich.print("==== SAMPLE ====")
#     rich.print(sample(output, total=10, n_positives=5))

# queries = np.random.randint(-1, dim, size=(bs, dim))
# keys = np.random.randint(-1, dim, size=(bs, dim))
# values = np.random.randn(bs, dim).astype(np.float32)
# rich.print(
#     {
#         "queries": queries,
#         "keys": keys,
#         "values": values,
#     }
# )
# output = gather_values_2d(queries, keys, values)
# rich.print(output)
