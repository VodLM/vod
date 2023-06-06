from __future__ import annotations

import copy
import dataclasses
import math
import string

import numpy as np
import pytest

from raffle_ds_research.core.mechanics.retrieval_collate import weighted_merge_search_results
from raffle_ds_research.tools import index_tools


@dataclasses.dataclass
class Task:
    """A task."""

    candidates: dict[str, index_tools.RetrievalBatch]
    weights: dict[str, float]


@pytest.fixture
def task(nkeys: int, batch_size: int, n_results: int, seed: int, id_range: int = 64) -> Task:
    """Generate a task."""
    rgn = np.random.RandomState(seed)
    candidates, weights = {}, {}
    for key in string.ascii_lowercase[:nkeys]:
        weight = rgn.randn()
        indices = np.stack([rgn.choice(id_range, n_results, replace=False) for _ in range(batch_size)], axis=0)
        scores = rgn.randn(batch_size, n_results)
        candidates[key] = index_tools.RetrievalBatch(indices=indices, scores=scores)
        weights[key] = weight
    return Task(candidates=candidates, weights=weights)


def _py_weighted_merge_search_results(
    candidates: dict[str, index_tools.RetrievalBatch], weights: dict[str, float]
) -> tuple[index_tools.RetrievalBatch, dict[str, np.ndarray]]:
    """Merge search results with weights."""

    def _search(index: int, indices: np.ndarray, scores: np.ndarray) -> np.nan | float:
        for i, s in zip(indices, scores):
            if i == index:
                return s

        return np.nan

    def _merge_single(
        candidates: dict[str, index_tools.RetrievalSample], weights: dict[str, float]
    ) -> tuple[index_tools.RetrievalSample, dict[str, np.ndarray]]:
        """Merge search samples."""
        new_indices, new_scores = [], []
        for key in candidates:
            candidate = candidates[key]
            weight = weights[key]
            for score, index in zip(candidate.scores, candidate.indices):
                weighted_score = weight * score
                if index in new_indices:
                    new_scores[new_indices.index(index)] += weighted_score
                else:
                    new_indices.append(index)
                    new_scores.append(weighted_score)

        merged_scores = {
            key: np.asarray([_search(i, candidates[key].indices, candidates[key].scores) for i in new_indices])
            for key in candidates
        }

        return (
            index_tools.RetrievalSample(
                indices=np.array(new_indices, dtype=np.int64),
                scores=np.array(new_scores, dtype=np.float32),
            ),
            merged_scores,
        )

    batch_size = next(iter(candidates.values())).indices.shape[0]
    merged_batch, merged_scores = [], {key: [] for key in candidates}
    for i in range(batch_size):
        merged_batch_i, merged_scores_i = _merge_single(
            {key: candidates[key][i] for key in candidates},
            weights=weights,
        )
        merged_batch.append(merged_batch_i)
        for key in candidates:
            merged_scores[key].append(merged_scores_i[key])

    # stack the results
    padded_size = max([r.indices.shape[0] for r in merged_batch])
    merged_batch = index_tools.RetrievalBatch(
        indices=np.stack(
            [
                np.pad(r.indices, (0, padded_size - r.indices.shape[0]), mode="constant", constant_values=-1)
                for r in merged_batch
            ],
            axis=0,
        ),
        scores=np.stack(
            [
                np.pad(r.scores, (0, padded_size - r.scores.shape[0]), mode="constant", constant_values=-math.inf)
                for r in merged_batch
            ],
            axis=0,
        ),
    )
    merged_scores = {
        key: np.stack(
            [
                np.pad(s, (0, padded_size - s.shape[0]), mode="constant", constant_values=-np.nan)
                for s in merged_scores[key]
            ],
            axis=0,
        )
        for key in merged_scores
    }

    return merged_batch, merged_scores


@pytest.mark.parametrize("nkeys", [2])
@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("n_results", [2, 32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_merge_search_results(task: Task) -> None:
    """Test `weighted_merge_search_results`."""
    ref_merged, ref_scores = _py_weighted_merge_search_results(
        copy.deepcopy(task.candidates),
        copy.deepcopy(task.weights),
    )
    merged, scores = weighted_merge_search_results(
        copy.deepcopy(task.candidates),
        copy.deepcopy(task.weights),
    )
    assert _all_close_no_nans(merged.indices, ref_merged.indices)
    assert _all_close_no_nans(merged.scores, ref_merged.scores)
    for key in scores:
        assert _all_close_no_nans(scores[key], ref_scores[key])


def _all_close_no_nans(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if two arrays are all close."""
    m = np.logical_or(np.isnan(a), np.isnan(b))
    u = 123456789.123456789  # <- a random number
    return np.allclose(
        np.where(m, u + np.zeros_like(a), a),
        np.where(m, u + np.zeros_like(b), b),
    )
