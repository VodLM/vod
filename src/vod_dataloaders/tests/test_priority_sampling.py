import collections

import numpy as np
import pytest
import rich
from vod_dataloaders.core import numpy_ops, sample


@pytest.mark.parametrize(
    "n_trials,atoal,n,k_positive,k_total,label_thres",
    [
        (10_000, 1e-2, 16, 4, 8, 0.5),
        (10_000, 1e-2, 16, 4, 8, 0),
        (10_000, 1e-2, 16, 4, 8, 1),
    ],
)
def test_labeled_priority_sampling(
    n_trials: int, atoal: float, n: int, k_positive: int, k_total: int, label_thres: float, inf_thres: float = 0.2
) -> None:
    """Generate a simple mean estimattion problem and check that the estimates are close to the true values."""
    f = np.random.normal(size=n)
    unorm_log_p = np.random.uniform(size=n)
    unorm_log_p[unorm_log_p < inf_thres] = -np.inf  # simulate some -inf scores
    labels = np.random.normal(size=n)
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

    rich.print(
        {
            "mu_a": mu_a,
            "mu_b": mu_b,
            "mu_a_hat": np.mean(mu_a_hats),
            "mu_b_hat": np.mean(mu_b_hats),
        }
    )

    if mu_a is not None:
        assert np.isclose(mu_a, np.mean(mu_a_hats), atol=atoal)
    if mu_b is not None:
        assert np.isclose(mu_b, np.mean(mu_b_hats), atol=atoal)
