import numpy as np
import pytest
from collections import defaultdict

from raffle_ds_research.tools import c_tools


@pytest.mark.parametrize("n_points", [10, 100, 1000])
@pytest.mark.parametrize("max_n_unique", [1, 10, 100])
@pytest.mark.parametrize("n_labels", [1, 3, 10])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_get_frequencies(n_points: int, max_n_unique: int, n_labels: int, seed: int):
    np.random.set_state(np.random.RandomState(seed).get_state())
    pids = np.random.randint(0, max_n_unique, size=(n_points,), dtype=np.uint64)
    labels = np.random.randint(0, n_labels, size=(n_points,), dtype=np.uint64)
    freqs = c_tools.get_frequencies(pids, labels=labels, n_labels=n_labels, max_n_unique=max_n_unique)
    _test_single(pids, labels, freqs, n_labels)


@pytest.mark.parametrize("batch_size", [1, 3, 10])
@pytest.mark.parametrize("n_points", [10, 100, 1000])
@pytest.mark.parametrize("max_n_unique", [1, 10, 100])
@pytest.mark.parametrize("n_labels", [1, 3, 10])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_get_frequencies_batched(batch_size: int, n_points: int, max_n_unique: int, n_labels: int, seed: int):
    np.random.set_state(np.random.RandomState(seed).get_state())
    pids = np.random.randint(
        0,
        max_n_unique,
        size=(
            batch_size,
            n_points,
        ),
        dtype=np.uint64,
    )
    labels = np.random.randint(
        0,
        n_labels,
        size=(
            batch_size,
            n_points,
        ),
        dtype=np.uint64,
    )
    freqs = c_tools.get_frequencies(pids, labels=labels, n_labels=n_labels, max_n_unique=max_n_unique)
    for i in range(batch_size):
        _test_single(pids[i], labels[i], freqs[i], n_labels)


def _test_single(pids: np.ndarray, labels: np.ndarray, freqs: c_tools.Frequencies, n_labels: int):
    upids = freqs.values
    ulabels = freqs.counts
    # remote the padding (set with -1)
    not_masked = upids >= 0
    upids = upids[not_masked]
    ulabels = ulabels[not_masked]

    # check that the output values match the original labels and pids.
    assert ulabels.shape == (
        upids.shape[0],
        n_labels,
    )
    if list(sorted(upids.tolist())) != list(sorted(set(upids.tolist()))):
        raise ValueError("The output upids are not unique.")
    if set(pids.tolist()) != set(upids.tolist()):
        raise ValueError(
            "The unique pids are not the same as the expected ones."
            "Found: {upids.tolist()}. Expected: {set(pids.tolist())}."
        )

    # build the simple Python data structures to compare the results
    expected_unique_pids_per_labels = defaultdict(set)
    for pid, label in zip(pids, labels):
        expected_unique_pids_per_labels[label].add(pid)
    found_unique_pids_per_labels = defaultdict(set)
    for upid, labels in zip(upids, ulabels):
        for label, state in enumerate(labels):
            if state:
                found_unique_pids_per_labels[int(label)].add(int(upid))

    # compare the results
    if set(expected_unique_pids_per_labels.keys()) != set(found_unique_pids_per_labels.keys()):
        raise ValueError("The labels are not the same as the expected ones.")
    for label in expected_unique_pids_per_labels.keys():
        if expected_unique_pids_per_labels[label] != found_unique_pids_per_labels[label]:
            raise ValueError(
                f"The unique pids are not the same as the expected ones. "
                f"Found: {found_unique_pids_per_labels[label]}. Expected: {expected_unique_pids_per_labels[label]}."
            )
