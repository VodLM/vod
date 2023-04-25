from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np
import pytest

from raffle_ds_research.tools import c_tools

DEBUG = False


@dataclasses.dataclass
class Expected:
    """Expected results from the C++ function."""

    indices: list[int]
    scores: list[float]
    labels: list[int]


def _py_sort_filter(
    indices: np.ndarray,
    scores: np.ndarray,
    exclude: Optional[list[int]] = None,
) -> tuple[list, list]:
    if exclude is None:
        exclude = []
    indices = [int(i) for i in indices]
    scores = [float(s) for s in scores]
    zipped = list((i, s) for i, s in zip(indices, scores) if i >= 0 and i not in exclude)
    zipped.sort(key=lambda x: x[1], reverse=True)
    indices, scores = zip(*zipped)
    return list(indices), list(scores)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("a_size", [10, 20, 30])
@pytest.mark.parametrize("b_size", [10, 20, 30])
@pytest.mark.parametrize("total", [10, 20, 30])
@pytest.mark.parametrize("max_a_size", [10, 20, 30])
@pytest.mark.parametrize("batched", [False])
def test_concat_topk(a_size: int, b_size: int, total: int, max_a_size: int, seed: int, batched: bool) -> None:
    """Test the `concat_topk` function."""
    rgn = np.random.RandomState(seed)
    a_indices = rgn.randint(0, a_size, size=(a_size,))
    a_scores = rgn.randn(a_size)
    b_indices = rgn.randint(200, 200 + b_size, size=(b_size,))
    b_scores = rgn.randn(b_size)

    # apply some padding
    n_a_padding = rgn.randint(0, a_size // 2)
    n_b_padding = rgn.randint(0, b_size // 2)
    a_indices = np.pad(a_indices, (0, n_a_padding), constant_values=-1)
    a_scores = np.pad(a_scores, (0, n_a_padding), constant_values=-math.inf)
    b_indices = np.pad(b_indices, (0, n_b_padding), constant_values=-1)
    b_scores = np.pad(b_scores, (0, n_b_padding), constant_values=-math.inf)

    if DEBUG:
        import rich

        rich.print(
            dict(
                a_indices=a_indices,
                a_scores=a_scores,
                b_indices=b_indices,
                b_scores=b_scores,
            )
        )

    # compute the expected result using a slow python implementation
    expected = _compute_expected(
        a_indices=a_indices,
        a_scores=a_scores,
        b_indices=b_indices,
        b_scores=b_scores,
        max_a_size=max_a_size,
        total=total,
    )

    # run the function
    concatenated: c_tools.ConcatenatedTopk = c_tools.concat_search_results(
        a_indices=a_indices.copy(),
        a_scores=a_scores.copy(),
        b_indices=b_indices.copy(),
        b_scores=b_scores.copy(),
        total=total,
        max_a=max_a_size,
    )

    if DEBUG:
        import rich

        rich.print(
            dict(
                concatenated_indices=concatenated.indices,
                expected_indices=expected.indices,
            )
        )

    # check the results
    _check_arrays(concatenated.indices, expected.indices, label=f"indices (total={total})", dtype=int)
    _check_arrays(concatenated.scores, expected.scores, label="scores", dtype=float)
    _check_arrays(concatenated.labels, expected.labels, label="labels", dtype=int)


def _compute_expected(
    *,
    a_indices: np.ndarray,
    a_scores: np.ndarray,
    b_indices: np.ndarray,
    b_scores: np.ndarray,
    max_a_size: int,
    total: int,
) -> Expected:
    a_indices_ = [int(i) for i in a_indices.copy().tolist()]
    a_indices, a_scores = _py_sort_filter(a_indices, a_scores)
    b_indices, b_scores = _py_sort_filter(b_indices, b_scores, exclude=a_indices_)
    max_a_size = min(max_a_size, total, len(a_indices))
    max_b_size = total - max_a_size
    expected_indices = a_indices[:max_a_size] + b_indices[:max_b_size]
    expected_scores = a_scores[:max_a_size] + b_scores[:max_b_size]
    expected_labels = [0] * len(a_indices[:max_a_size]) + [1] * len(b_indices[:max_b_size])
    return Expected(
        indices=expected_indices,
        scores=expected_scores,
        labels=expected_labels,
    )


def _check_arrays(found: list | np.ndarray, expected: list | np.ndarray, *, label: str, dtype: type) -> None:
    if isinstance(found, np.ndarray):
        found = found.tolist()
    found = [dtype(x) for x in found]

    if isinstance(expected, np.ndarray):
        expected = expected.tolist()
    expected = [dtype(x) for x in expected]

    if len(found) != len(expected):
        raise AssertionError(
            f"Attribute `{label}`: length mismatch. Found: {len(found)}, expected: {len(expected)}. "
            f"Found: {found}, expected: {expected}"
        )

    mismatches = [i for i, (x, y) in enumerate(zip(found, expected)) if not np.isclose(x, y)]
    if len(mismatches) > 0:
        mismatches_str = [f"{i}: {found[i]} != {expected[i]}" for i in mismatches]
        raise AssertionError(f"Attribute `{label}`: value mismatch on {mismatches_str}")
