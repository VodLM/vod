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
cdef void _gather_by_index(
    DTYPE_LONG [:] queries,
    DTYPE_LONG [:] keys,
    DTYPE_FLOAT [:] values,
    DTYPE_FLOAT [:] buffer,
) nogil:

    cdef:
        unsigned long i, j
        DTYPE_LONG buffered_query

    for i in range(len(queries)):
        buffered_query = queries[i]
        if buffered_query < 0:
            continue
        for j in range(len(keys)):
            if buffered_query == keys[j]:
                buffer[i] = values[j]
                break


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void  _gather_by_index_batch(
        DTYPE_LONG [:,:] queries,
        DTYPE_LONG [:,:] keys,
        DTYPE_FLOAT [:,:] values,
        DTYPE_FLOAT [:,:] buffer,
) nogil:
    cdef unsigned long i
    cdef unsigned long batch_size = queries.shape[0]
    for i in prange(batch_size, nogil=True):
        _gather_by_index(
            queries[i],
            keys[i],
            values[i],
            buffer[i],
        )


def gather_by_index(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Gather values from `values` indexed by `queries` using `keys` as a lookup table.
    The `keys` array is expected to have unique values, the first match is returned.
    """

    assert keys.shape == values.shape
    dims = len(queries.shape)
    if any([d != dims for d in [len(queries.shape), len(keys.shape), len(values.shape)]]):
        raise ValueError(f"Input arrays must have the same number of dimensions. "
                         f"Found: {queries.shape}, {keys.shape}, {values.shape}.")

    # casting
    queries = queries.astype(NP_DTYPE_LONG)
    keys = keys.astype(NP_DTYPE_LONG)
    values = values.astype(NP_DTYPE_FLOAT)
    buffer = np.full(queries.shape, fill_value, dtype=NP_DTYPE_FLOAT)

    # run the C function
    if dims == 1:
        _gather_by_index(queries, keys, values, buffer)
    elif dims == 2:
        _gather_by_index_batch(queries, keys, values, buffer)
    else:
        raise ValueError(f"Only 1D and 2D arrays are supported. Found: {dims}D array.")

    return buffer
