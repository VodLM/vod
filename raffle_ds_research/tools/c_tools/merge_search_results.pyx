import math
from typing import Tuple

cimport cython
import numpy as np
from cython.parallel cimport prange
from typing_extensions import TypeAlias

ctypedef long long DTYPE_LONG
ctypedef float DTYPE_FLOAT
NP_DTYPE_LONG: TypeAlias = np.int64
NP_DTYPE_FLOAT: TypeAlias = np.float32


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_LONG _merge_search_results(
    DTYPE_LONG [:] ab_indices,
    DTYPE_FLOAT [:, :] ab_scores,
    DTYPE_LONG [:] a_indices,
    DTYPE_FLOAT [:] a_scores,
    DTYPE_LONG [:] b_indices,
    DTYPE_FLOAT [:] b_scores,
) nogil:

    cdef:
        unsigned long i, j, k
        DTYPE_LONG cursor = 0
        DTYPE_LONG buffered_idx_a
        DTYPE_LONG buffered_idx_b
        DTYPE_LONG idx_found

    # fill the `ab_indices` with values from `a`
    for i in range(len(a_indices)):
        buffered_idx_a = a_indices[i]
        if buffered_idx_a < 0:
            continue

        # register the value of `a`
        ab_indices[cursor] = buffered_idx_a
        ab_scores[cursor, 0] = a_scores[i]
        cursor += 1

    # fill the `ab_indices` with values from `b`
    for j in range(len(b_indices)):
        buffered_idx_b = b_indices[j]
        if buffered_idx_b < 0:
            continue

        # scan the `ab_indices` to see if we have already seen this index
        idx_found = -1
        for k in range(len(ab_indices)):
            if ab_indices[k] == buffered_idx_b:
                idx_found = k
                break

        # register the value of b
        if idx_found >= 0:
            ab_scores[idx_found, 1] = b_scores[j]
        else:
            ab_indices[cursor] = buffered_idx_b
            ab_scores[cursor, 1] = b_scores[j]
            cursor += 1

    return cursor


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_LONG [:] _merge_search_results_batch(
    DTYPE_LONG [:, :] ab_indices,
    DTYPE_FLOAT [:, :, :] ab_scores,
    DTYPE_LONG [:, :] a_indices,
    DTYPE_FLOAT [:, :] a_scores,
    DTYPE_LONG [:, :] b_indices,
    DTYPE_FLOAT [:, :] b_scores,
    DTYPE_LONG [:] cursors,
) nogil:
    cdef unsigned long i
    cdef unsigned long batch_size = ab_indices.shape[0]
    for i in prange(batch_size, nogil=True):
        cursors[i] = _merge_search_results(
            ab_indices[i],
            ab_scores[i],
            a_indices=a_indices[i],
            a_scores=a_scores[i],
            b_indices=b_indices[i],
            b_scores=b_scores[i],
        )

    return cursors


def merge_search_results(
    *,
    a_indices: np.ndarray,
    a_scores: np.ndarray,
    b_indices: np.ndarray,
    b_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """warning: assumes `a_indices` and `b_indices` to be unique."""
    assert a_indices.shape == a_scores.shape
    assert b_indices.shape == b_scores.shape
    dim = a_indices.ndim
    n_total = a_indices.shape[-1] + b_indices.shape[-1]
    if dim> 2:
        raise ValueError("a_indices and b_indices must be 1D or 2D")

    if dim == 1:
        new_indices = np.full(n_total, dtype=NP_DTYPE_LONG, fill_value=-1)
        new_scores = np.full((n_total, 2), dtype=NP_DTYPE_FLOAT, fill_value=np.nan)
        cursor = _merge_search_results(
            new_indices,
            new_scores,
            a_indices=a_indices.astype(NP_DTYPE_LONG),
            a_scores = a_scores.astype(NP_DTYPE_FLOAT),
            b_indices = b_indices.astype(NP_DTYPE_LONG),
            b_scores = b_scores.astype(NP_DTYPE_FLOAT),
        )
        new_indices = new_indices[:cursor]
        new_scores = new_scores[:cursor]
        return new_indices, new_scores
    elif dim == 2:
        batch_size = a_indices.shape[0]
        assert batch_size == b_indices.shape[0]
        new_indices = np.full((batch_size, n_total), dtype=NP_DTYPE_LONG, fill_value=-1)
        new_scores = np.full((batch_size, n_total, 2), dtype=NP_DTYPE_FLOAT, fill_value=np.nan)
        cursors = _merge_search_results_batch(
            new_indices,
            new_scores,
            a_indices=a_indices.astype(NP_DTYPE_LONG),
            a_scores=a_scores.astype(NP_DTYPE_FLOAT),
            b_indices=b_indices.astype(NP_DTYPE_LONG),
            b_scores=b_scores.astype(NP_DTYPE_FLOAT),
            cursors=np.full(batch_size, dtype=NP_DTYPE_LONG, fill_value=-1),
        )
        cursors = np.asarray(cursors)
        cursor = cursors.max()
        new_indices = new_indices[:, :cursor]
        new_scores = new_scores[:, :cursor, :]
        return new_indices, new_scores
    else:
        raise ValueError("a_indices and b_indices must be 1D or 2D")





