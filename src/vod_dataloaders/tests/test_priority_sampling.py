import collections

import numpy as np
import pytest
from vod_dataloaders.core import numpy_ops, sample


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "n_trials,n,k,inf_frac",
    [
        (1_00, 100, 10, 0),
        (1_000, 100, 10, 0),
        (1_00, 100, 100, 0),
        (1_000, 100, 10, 0.5),
        (1_000, 100, 10, 95),
    ],
)
def test_priority_sampling_1d(seed: int, n_trials: int, n: int, k: int, dtype: str, inf_frac: float) -> None:
    """Generate a simple mean estimattion problem and check that the estimates are close to the true values."""
    rgn = np.random.default_rng(seed)
    f = rgn.normal(size=n).astype(dtype)
    unorm_log_p = rgn.uniform(size=n).astype(dtype)

    # randomly set some scores to -inf, but make sure at least one score is not -inf
    if inf_frac > 0:
        unorm_log_p[rgn.uniform(size=n) < inf_frac] = -np.inf
    if np.all(unorm_log_p == -np.inf):  # if all scores are -inf, re-init some scores
        m = rgn.uniform(size=n) < (1 - inf_frac)
        unorm_log_p = np.where(m, unorm_log_p, rgn.normal(size=len(unorm_log_p)))

    # targets
    mu = np.sum(numpy_ops.softmax_1d(unorm_log_p) * f)

    # estimates
    mu_hats = []
    for _ in range(n_trials):
        z, log_w = sample.priority_sampling_1d(unorm_log_p, k=k)
        assert np.all(~np.isnan(log_w))
        assert z.dtype == np.int64
        assert log_w.dtype == unorm_log_p.dtype
        mu_hats += [np.sum(numpy_ops.softmax_1d(log_w) * np.take(f, z))]

    # Error bound: O(1/sqrt(n_trials * k))
    atol = 10.0 / (np.sqrt(n_trials * k))
    assert np.isclose(mu, np.mean(mu_hats), atol=atol)


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize(
    "n_trials,n,k_positive,k_total,label_thres",
    [
        (3_000, 32, 4, 8, 0.5),
        (3_000, 32, 4, 8, 0),
        (3_000, 32, 4, 8, 1),
    ],
)
def test_labeled_priority_sampling(
    seed: int,
    n_trials: int,
    n: int,
    k_positive: int,
    k_total: int,
    label_thres: float,
    inf_thres: float = 0.2,
    dtype: str = "float32",
) -> None:
    """Generate a simple mean estimattion problem and check that the estimates are close to the true values."""
    rgn = np.random.default_rng(seed)
    f = rgn.normal(size=n).astype(dtype)
    unorm_log_p = rgn.uniform(size=n).astype(dtype)
    unorm_log_p[unorm_log_p < inf_thres] = -np.inf  # simulate some -inf scores
    labels = rgn.normal(size=n)
    labels = np.where(labels > label_thres, 1, 0)

    # targets
    mu_a = np.sum(numpy_ops.softmax_1d(unorm_log_p[labels == 1]) * f[labels == 1]) if np.sum(labels == 1) > 0 else None
    mu_b = np.sum(numpy_ops.softmax_1d(unorm_log_p[labels == 0]) * f[labels == 0]) if np.sum(labels == 0) > 0 else None

    mu_a_hats, mu_b_hats = [], []
    z_, log_w_, ls_ = sample.labeled_priority_sampling(
        unorm_log_p[None].repeat(n_trials, axis=0),
        labels[None].repeat(n_trials, axis=0),
        k_positive=k_positive,
        k_total=k_total,
        normalized=False,
    )
    assert np.all(~np.isnan(log_w_))
    for i in range(n_trials):
        z, log_w, ls = z_[i], log_w_[i], ls_[i]

        # Check that A and B are disjoint
        counts = collections.Counter(z)
        assert max(counts.values()) == 1

        # Store estimates
        if mu_a is not None:
            mu_a_hats += [np.sum(numpy_ops.softmax_1d(log_w[ls == 1]) * np.take(f, z[ls == 1]))]
        if mu_b is not None:
            mu_b_hats += [np.sum(numpy_ops.softmax_1d(log_w[ls == 0]) * np.take(f, z[ls == 0]))]

    if mu_a is not None:
        # Error bound: O(1/sqrt(n_trials * k_positive))
        pos_atol = 10.0 / (np.sqrt(n_trials * min(k_positive, np.sum(labels == 1))))
        assert np.isclose(mu_a, np.mean(mu_a_hats), atol=pos_atol)
    if mu_b is not None:
        # Error bound: O(1/sqrt(n_trials * (k_total - k_positive)))
        neg_atol = 10.0 / (np.sqrt(n_trials * min(k_total - k_positive, np.sum(labels == 0))))
        assert np.isclose(mu_b, np.mean(mu_b_hats), atol=neg_atol)
