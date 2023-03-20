cimport cython

from typing import Union

import numpy as np
import torch
from cython.parallel import prange
from typing_extensions import TypeAlias

ctypedef long long DTYPE_LONG
ctypedef float DTYPE_FLOAT
NP_DTYPE_LONG: TypeAlias = np.int64
NP_DTYPE_FLOAT: TypeAlias = np.float32
PyArray: TypeAlias = Union[np.ndarray, list, torch.Tensor]

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void _copy_by_index(
    DTYPE_LONG [:] a_indices,
    DTYPE_FLOAT [:] a_values,
    DTYPE_LONG [:] b_indices,
    DTYPE_FLOAT [:] b_values,
) nogil:

    cdef:
        unsigned long i, j
        unsigned int found_in_a = 0
        DTYPE_LONG buffered_idx_a

    for i in range(len(a_indices)):
        buffered_idx_a = a_indices[i]
        for j in range(len(b_indices)):
            if buffered_idx_a == b_indices[j]:
                a_values[i] = b_values[j]
                break


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void  _copy_by_index_batch(
    DTYPE_LONG [:, :] a_indices,
    DTYPE_FLOAT [:, :] a_values,
    DTYPE_LONG [:, :] b_indices,
    DTYPE_FLOAT [:, :] b_values,
) nogil:
    cdef unsigned long i
    cdef unsigned long batch_size = a_indices.shape[0]
    for i in prange(batch_size, nogil=True):
        _copy_by_index(
            a_indices[i],
            a_values[i],
            b_indices[i],
            b_values[i],
        )


def copy_by_index(
    a_indices: np.ndarray,
    a_values: np.ndarray,
    b_indices: np.ndarray,
    b_values: np.ndarray,
) -> np.ndarray:
    """
    Copy values from `b` to `a` by index.
    todo: check that a_indices and b_indices are unique. This is not checked and assumed.
    """

    a_values_ = a_values.astype(NP_DTYPE_FLOAT, copy=True)
    assert a_values.shape == a_indices.shape
    assert b_values.shape == b_indices.shape
    dims = len(a_indices.shape)
    if any([d != dims for d in [len(a_values.shape), len(b_indices.shape), len(b_values.shape)]]):
        raise ValueError(f"Input arrays must have the same number of dimensions. "
                         f"Found: {a_indices.shape}, {a_values.shape}, {b_indices.shape}, {b_values.shape}")

    if dims == 1:
        _copy_by_index(
            a_indices.astype(NP_DTYPE_LONG),
            a_values_,
            b_indices.astype(NP_DTYPE_LONG),
            b_values.astype(NP_DTYPE_FLOAT),
        )
    elif dims == 2:
        _copy_by_index_batch(
            a_indices.astype(NP_DTYPE_LONG),
            a_values_,
            b_indices.astype(NP_DTYPE_LONG),
            b_values.astype(NP_DTYPE_FLOAT),
        )
    else:
        raise ValueError(f"Only 1D and 2D arrays are supported. Found: {dims}D array.")

    return a_values_
