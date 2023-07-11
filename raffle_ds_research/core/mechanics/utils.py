from __future__ import annotations

import re
import time
import warnings
from typing import Any

import numpy as np


def numpy_gumbel_like(x: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    """Sample Gumbel(0, 1) random variables."""
    noise = np.random.uniform(low=eps, high=1.0 - eps, size=x.shape)
    return -np.log(-np.log(noise))


def numpy_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/special/_logsumexp.py#L228."""
    is_nan = np.isnan(x)
    x_safe = np.where(is_nan, -np.inf, x)
    x_max = np.amax(x_safe, axis=axis, keepdims=True)

    if x_max.ndim > 0:
        x_max[np.isinf(x_max) | np.isnan(x_max)] = 0
    elif np.isinf(x_max) | np.isnan(x_max):
        x_max = 0

    x_minus_max = x - x_max
    exp_x_minus_max = np.exp(x_minus_max)

    # suppress warnings about log of zero
    sum_exp_x = np.sum(exp_x_minus_max, axis=axis, keepdims=True)
    lse_x = np.log(sum_exp_x)

    lse_x = exp_x_minus_max - lse_x
    return np.where(is_nan, np.nan, lse_x)


def fill_nans_with_min(values: np.ndarray, offset_min_value: None | float = -1, axis: int = -1) -> np.ndarray:
    """Replace NaNs with the minimum value along each dimension."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        min_scores = np.nanmin(values, axis=axis, keepdims=True)
        min_scores = np.where(np.isnan(min_scores), 0, min_scores)
        if offset_min_value is not None:
            min_scores += offset_min_value  # make sure the min is lower than the rest
        return np.where(np.isnan(values), min_scores, values)


_clean_query_ptrn = re.compile(r"^(q:|query:|question:|question\s+text:|question\s+text\s+is:)\s*", re.IGNORECASE)
_strip_punctuation_ptrn = re.compile(r"[^\w\s]")
_subsequent_whitespaces_ptrn = re.compile(r"\s+")


def exact_match(q: str, d: str) -> bool:
    """Exact match between query and document."""
    q = _clean_query_ptrn.sub("", q)
    q = _strip_punctuation_ptrn.sub(" ", q).lower()
    d = _strip_punctuation_ptrn.sub(" ", d).lower()
    q = _subsequent_whitespaces_ptrn.sub(" ", q)
    d = _subsequent_whitespaces_ptrn.sub(" ", d)
    import rich

    rich.print({"q": q, "d": d})
    return q in d


class BlockTimer:
    """A context manager for timing code blocks."""

    def __init__(self, name: str, output: dict[str, Any]) -> None:
        self.name = name
        self.output = output

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, *args: Any) -> None:
        self.output[self.name] = time.perf_counter() - self.start
