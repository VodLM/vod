import numpy as np
import pytest

from raffle_ds_research.tools import c_tools


def _get_expected_labels(labels_a, scores_a, labels_b, scores_b):
    new_scores_a = scores_a.copy()
    for i, la in enumerate(labels_a):
        for j, lb in enumerate(labels_b):
            if la == lb:
                new_scores_a[i] = scores_b[j]
                break
    return new_scores_a


@pytest.mark.parametrize("n_points", [10, 100])
@pytest.mark.parametrize("max_n_unique", [10, 100])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_get_frequencies(n_points: int, max_n_unique: int, seed: int):
    np.random.set_state(np.random.RandomState(seed).get_state())
    labels_a = np.random.randint(0, max_n_unique, size=(n_points,), dtype=np.uint64)
    labels_b = np.random.randint(0, max_n_unique, size=(n_points,), dtype=np.uint64)
    scores_a = np.random.random(size=(n_points,))
    scores_b = np.random.random(size=(n_points,))

    new_scores_a = c_tools.copy_by_index(labels_a, scores_a, labels_b, scores_b)
    expected_scores_a = _get_expected_labels(labels_a, scores_a, labels_b, scores_b)
    for i, (na, ea) in enumerate(zip(new_scores_a, expected_scores_a)):
        if not np.isclose(na, ea):
            raise ValueError(
                f"new_scores_a[{i}] = {na} != {ea} = expected_scores_a[{i}]"
                f" (labels_a[{i}] = {labels_a[i]}, labels_b[{i}] = {labels_b[i]})"
            )
