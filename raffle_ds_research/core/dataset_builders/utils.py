from __future__ import annotations

import numpy as np


def numpy_gumbel_like(x: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    """Sample Gumbel(0, 1) random variables."""
    noise = np.random.uniform(low=eps, high=1.0 - eps, size=x.shape)
    return -np.log(-np.log(noise))


def numpy_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/special/_logsumexp.py#L228"""
    x_max = np.amax(x, axis=axis, keepdims=True)

    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0

    tmp = x - x_max
    exp_tmp = np.exp(tmp)

    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out = np.log(s)

    out = tmp - out
    return out