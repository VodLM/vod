import numpy as np
import pytest
import vod_types as vt
from vod_dataloaders.core.merge import merge_search_results


@pytest.fixture
def search_results(seed: int, seq_length: int, n_values: int) -> dict[str, vt.RetrievalBatch]:
    rgn = np.random.default_rng(seed)
    alen = seq_length // 2
    blen = seq_length - alen

    # assign labels for each index value
    ids = np.arange(n_values)
    labels = {i: rgn.choice([False, True], p=[0.5, 0.5]) for i in ids}

    # Sample indices without replacement
    a_indices = rgn.choice(ids, size=(alen,), replace=False)
    b_indices = rgn.choice(ids, size=(blen,), replace=False)

    return {
        "a": vt.RetrievalBatch.cast(
            indices=a_indices[None, :],
            labels=[[labels[i] for i in a_indices]],
            scores=rgn.uniform(
                0.0,
                10.0,
                size=(1, alen),
            ),
        ),
        "b": vt.RetrievalBatch.cast(
            indices=b_indices[None, :],
            labels=[[labels[i] for i in b_indices]],
            scores=rgn.uniform(
                0.0,
                10.0,
                size=(1, blen),
            ),
        ),
    }


@pytest.fixture
def weights(seed: int) -> dict[str, float]:
    rgn = np.random.default_rng(seed)
    return {
        "a": rgn.uniform(0.0, 1.0),
        "b": rgn.uniform(0.0, 1.0),
    }


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("seq_length", [10, 30])
@pytest.mark.parametrize("n_values", [300, 1000])
def test_merge_search_results(search_results: dict[str, vt.RetrievalBatch], weights: dict[str, float]) -> None:
    """Test merging multiple search results.

    Expected the output scores to be the weighted sum of the input scores.
    """
    merged, raw_scores = merge_search_results(search_results, weights)

    # Build lookup dictionaries for the input values
    input_scores_lookups: dict[str, dict[int, float]] = {
        key: dict(zip(v.indices[0], v.scores[0])) for key, v in search_results.items()
    }

    # Test that the values of the raw scores are the same as the input scores
    # When a index was not present in the input scores, the raw score should be `np.nan`
    indices = merged.indices[0]
    for key, raw_scores_key in raw_scores.items():
        for i, s_raw in zip(indices, raw_scores_key[0]):
            s_input = input_scores_lookups[key].get(i, np.nan)
            assert (np.isnan(s_input) and np.isnan(s_raw)) or (s_raw == s_input)

    # Test the value of the merged scores, it should be equal to the weighted sum of the input scores
    for i, merged_s in zip(indices, merged.scores[0]):
        if i < 0:
            assert merged_s == -np.inf
            continue
        expected_value = sum(input_scores_lookups[key].get(i, 0.0) * weight for key, weight in weights.items())
        assert merged_s == expected_value
