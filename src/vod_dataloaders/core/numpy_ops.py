import typing as typ
import warnings

import numba
import numpy as np
import vod_types as vt
from numpy import typing as npt

CACHE_NUMBA_JIT = True

Dtyp = typ.TypeVar("Dtyp", bound=np.float_ | np.int_ | np.bool_)
Float = typ.TypeVar("Float", bound=np.float_)
Bool = typ.TypeVar("Bool", bound=np.bool_)
Int = typ.TypeVar("Int", bound=np.int_)


def _default_fill_value(dt: np.dtype) -> float | int:
    """Return the default fill value for a given dtype."""
    if dt.kind == "f":
        return np.nan

    return -1


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True)
def _nopy_gather_values_1d(
    queries: npt.NDArray[Int],
    keys: npt.NDArray[Int],
    values: npt.NDArray[Dtyp],
    output: npt.NDArray[Dtyp],
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
    queries: npt.NDArray[Int],
    keys: npt.NDArray[Int],
    values: npt.NDArray[Dtyp],
    fill_value: typ.Optional[float | int] = None,
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


@numba.njit(parallel=True, fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True)
def _nopy_gather_values_2d(
    queries: npt.NDArray[Int],
    keys: npt.NDArray[Int],
    values: npt.NDArray[Dtyp],
    output: npt.NDArray[Dtyp],
) -> None:
    bs = len(queries)
    if len(keys) != bs:
        raise ValueError(f"Expected keys to have length {bs}. Found: {len(keys)}")
    if len(values) != bs:
        raise ValueError(f"Expected values to have length {bs}. Found: {len(values)}")

    for i in numba.prange(bs):
        _nopy_gather_values_1d(queries[i], keys[i], values[i], output[i])


@numba.njit(parallel=True, fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True)
def _nopy_gather_values_2d_from_1d(
    queries: npt.NDArray[Int],
    keys: npt.NDArray[Int],
    values: npt.NDArray[Dtyp],
    output: npt.NDArray[Dtyp],
) -> None:
    bs = len(queries)
    if keys.ndim != 1:
        raise ValueError(f"Expected keys to have ndim 1. Found: {keys.ndim}")
    if values.ndim != 1:
        raise ValueError(f"Expected values to have ndim 1. Found: {values.ndim}")

    for i in numba.prange(bs):
        _nopy_gather_values_1d(queries[i], keys, values, output[i])


def gather_values_2d(
    queries: npt.NDArray[Int],
    keys: npt.NDArray[Int],
    values: npt.NDArray[Dtyp],
    fill_value: typ.Optional[float | int] = None,
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
    queries: npt.NDArray[Int],
    keys: npt.NDArray[Int],
    values: npt.NDArray[Dtyp],
    fill_value: typ.Optional[float | int] = None,
) -> npt.NDArray[Dtyp]:
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
    queries: npt.NDArray[Int],
    indices: npt.NDArray[Int],
    values: npt.NDArray[Dtyp],
    fill_value: typ.Optional[float | int] = None,
) -> npt.NDArray[Dtyp]:
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


def log_softmax(x: npt.NDArray[Float], dim: int = -1) -> npt.NDArray[Float]:
    """Compute log-softmax values for a given tensor."""
    x_safe = np.where(np.isnan(x), -np.inf, x)

    # substract the max
    x_max = np.max(x_safe, axis=dim, keepdims=True)
    x_max[np.isinf(x_max)] = x.dtype.type(0)
    x_safe = x_safe - x_max

    # compute the log sum exp
    lse = np.log(np.sum(np.exp(x_safe), axis=dim, keepdims=True))

    # compute the log softmax
    return x_safe - lse


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True)
def max_1d(x_safe: npt.NDArray[Float], default_value: float = 0.0) -> Float:
    """Compute the max value of a 1d array."""
    xm = x_safe.dtype.type(-np.inf)
    for v in x_safe:
        if v > xm:
            xm = v

    if not np.isfinite(xm) and xm < 0:
        xm = x_safe.dtype.type(default_value)

    return xm


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True, parallel=True)
def masked_fill_1d_(mask: npt.NDArray[Bool], x: npt.NDArray[Float], fill_value: Float) -> None:
    """Fill values in a 1d array based on a condition."""
    for i in range(len(x)):
        if mask[i]:
            x[i] = fill_value


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True, parallel=True)
def mul_1d_(x: npt.NDArray[Float], value: Float) -> None:
    """Multiply values in a 1d array by a constant."""
    for i in range(len(x)):
        x[i] *= value


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True, parallel=True)
def add_1d_(x: npt.NDArray[Float], value: Float) -> None:
    """Substract a constant from values in a 1d array."""
    for i in range(len(x)):
        x[i] += value


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True, parallel=True)
def _logsumexp_1d(x: npt.NDArray[Float]) -> Float:
    lse = x.dtype.type(0.0)
    for v in x:
        lse += np.exp(v)

    return np.log(lse)


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True)
def log_softmax_1d_(x: npt.NDArray[Float]) -> None:
    """Compute log-softmax values for a given tensor."""
    masked_fill_1d_(np.isnan(x), x, x.dtype.type(-np.inf))

    # substract the max
    add_1d_(x, -max_1d(x))

    # compute the log sum exp and substract it
    add_1d_(x, -_logsumexp_1d(x))


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True)
def log_softmax_1d(x: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """Compute log-softmax values for a given tensor."""
    x = x.copy()
    log_softmax_1d_(x)
    return x


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True)
def softmax_1d_(x: npt.NDArray[Float]) -> None:
    """Compute softmax values for a given tensor."""
    log_softmax_1d_(x)
    np.exp(x, x)  # inplace exp


@numba.njit(fastmath=True, cache=CACHE_NUMBA_JIT, nogil=True)
def softmax_1d(x: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """Compute softmax values for a given tensor."""
    x = x.copy()
    softmax_1d_(x)
    return x


def fill_nans_with_min(
    values: npt.NDArray[Float],
    offset_min_value: None | float = -1,
    axis: int = -1,
) -> npt.NDArray[Float]:
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
