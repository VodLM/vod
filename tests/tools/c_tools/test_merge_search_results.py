import math

import numpy as np
import pytest
import rich

from raffle_ds_research.tools import c_tools, index_tools


def gen_batch(batch_size: int, n_points: int, n_range, seed: int):
    rgn = np.random.RandomState(seed)
    examples = []
    for _ in range(batch_size):
        # generate a sequence of n_points integers in the range [0, n_range)
        indices = rgn.randint(0, n_range, size=n_points)
        indices = np.unique(indices)
        # pad the sequence to length n_points
        indices = np.pad(indices, (0, n_points - indices.size), mode="constant", constant_values=-1)
        scores = rgn.rand(n_points)
        scores[indices == -1] = -math.inf
        examples.append(dict(indices=indices, scores=scores))
    indices = np.stack([ex["indices"] for ex in examples], axis=0)
    scores = np.stack([ex["scores"] for ex in examples], axis=0)
    return index_tools.RetrievalBatch(scores=scores, indices=indices)


@pytest.fixture
def batch_a(batch_size: int, na: int, n_range: int, seed: int):
    return gen_batch(batch_size, na, n_range, seed)


@pytest.fixture
def batch_b(batch_size: int, nb: int, n_range: int, seed: int):
    return gen_batch(batch_size, nb, n_range, seed + 1)


@pytest.mark.parametrize("na", [3, 10])
@pytest.mark.parametrize("nb", [3, 10])
@pytest.mark.parametrize("n_range", [3, 10, 100])
@pytest.mark.parametrize("batch_size", [1, 3, 100])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gather_by_index(batch_a, batch_b):
    merged_indices, merged_scores = c_tools.merge_search_results(
        a_indices=batch_a.indices,
        a_scores=batch_a.scores,
        b_indices=batch_b.indices,
        b_scores=batch_b.scores,
    )

    # check that the merged indices are sorted
    for i in range(merged_indices.shape[0]):
        merged_indices_i = [int(x) for x in merged_indices[i]]
        assert set(merged_indices_i) - {-1} == set(batch_a.indices[i]).union(set(batch_b.indices[i])) - {-1}

        # check batch A
        a_indices_i = batch_a.indices[i]
        a_scores_i = batch_a.scores[i]
        merged_scores_i_a = [float(x) for x in merged_scores[i, :, 0]]
        for a_i, a_s in zip(a_indices_i, a_scores_i):
            if a_i < 0:
                continue
            assert a_i in merged_indices_i
            a_i_j = merged_indices_i.index(a_i)
            assert np.allclose(merged_scores_i_a[a_i_j], a_s)

        # check batch B
        b_indices_i = batch_b.indices[i]
        b_scores_i = batch_b.scores[i]
        merged_scores_i_b = [float(x) for x in merged_scores[i, :, 1]]
        for b_i, b_s in zip(b_indices_i, b_scores_i):
            if b_i < 0:
                continue
            assert b_i in merged_indices_i
            b_i_j = merged_indices_i.index(b_i)
            assert np.allclose(merged_scores_i_b[b_i_j], b_s)
