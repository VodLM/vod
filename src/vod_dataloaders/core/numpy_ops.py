import warnings
from typing import Optional

import numba
import numpy as np
import vod_types as vt
from numpy import typing as npt

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


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT)
def log_softmax_1d(x: npt.NDArray) -> npt.NDArray:
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
def softmax_1d(x: npt.NDArray) -> npt.NDArray:
    """Compute softmax values for a given tensor."""
    return np.exp(log_softmax_1d(x))


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT)
def gumbel_from_uniform(u: npt.NDArray, eps: float = 1e-18) -> npt.NDArray:
    """Sample Gumbel(0, 1) random variables."""
    u = eps + (1 - 2 * eps) * u
    return -np.log(-np.log(u))


def fill_nans_with_min(values: np.ndarray, offset_min_value: None | float = -1, axis: int = -1) -> np.ndarray:
    """Replace NaNs with the minimum value along each dimension."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        min_scores = np.nanmin(values, axis=axis, keepdims=True)
        min_scores = np.where(np.isnan(min_scores), 0, min_scores)
        if offset_min_value is not None:
            min_scores += offset_min_value  # make sure the min is lower than the rest
        return np.where(np.isnan(values), min_scores, values)


def replace_negative_indices(sections: vt.RetrievalBatch, world_size: int) -> vt.RetrievalBatch:
    """Replace negative indices with random ones."""
    is_negative = sections.indices < 0
    n_negative = is_negative.sum()
    if n_negative:
        sections.indices.setflags(write=True)
        sections.indices[is_negative] = np.random.randint(0, world_size, size=n_negative)
    return sections
