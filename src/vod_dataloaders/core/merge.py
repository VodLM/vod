import numba
import numpy as np
import numpy.typing as npt
import vod_types as vt
from vod_dataloaders.core import numpy_ops


def merge_search_results(
    search_results: dict[str, vt.RetrievalBatch],
    weights: None | dict[str, float] = None,
) -> tuple[vt.RetrievalBatch, dict[str, npt.NDArray]]:
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


def _merge_n_search_results(
    search_results: dict[str, vt.RetrievalBatch],
    weights: dict[str, float],
) -> tuple[vt.RetrievalBatch, dict[str, npt.NDArray]]:
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
        raw_scores[key] = numpy_ops.gather_values_by_indices(
            queries=output.indices, indices=value.indices, values=value.scores
        )

    # lookup labels
    output.labels = None
    for r in search_results.values():
        if r.labels is not None:
            output.labels = numpy_ops.gather_values_by_indices(
                queries=output.indices,
                indices=r.indices,
                values=r.labels,
                fill_value=-1,
            )

    return output, raw_scores


def _merge_two_search_results(a: vt.RetrievalBatch, b: vt.RetrievalBatch) -> vt.RetrievalBatch:
    """Merge two search results."""
    scores, indices = _nopy_merge_two_search_results(a.scores, a.indices, b.scores, b.indices)
    return vt.RetrievalBatch(scores=scores, indices=indices)


@numba.njit(fastmath=True)
def _search_1d_arr(arr: npt.NDArray, x: int | float) -> int:
    """Search for values in a sorted array."""
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")

    for i, v in enumerate(arr):
        if v == x:
            return i

    return -1


@numba.njit(fastmath=True, cache=numpy_ops.CACHE_NUMBA_JIT)
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


@numba.njit(parallel=True, fastmath=True, cache=numpy_ops.CACHE_NUMBA_JIT)
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
