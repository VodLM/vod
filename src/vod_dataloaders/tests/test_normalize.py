import numpy as np
import pytest
from vod_dataloaders.core import normalize


@pytest.fixture
def scores(seed: int, nan_prob: float, inf_prob: float, n: int = 100) -> np.ndarray:
    rgn = np.random.default_rng(seed)
    scores = rgn.uniform(0.0, 10.0, size=(n,))
    scores = np.where(rgn.uniform(0.0, 1.0, size=(n,)) < nan_prob, np.nan, scores)
    scores = np.where(rgn.uniform(0.0, 1.0, size=(n,)) < inf_prob, -np.inf, scores)
    return scores


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("nan_prob", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("inf_prob", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("offset", [0, 1.0, -10.0])
def test_substract_min_score(
    scores: np.ndarray,
    offset: float,
) -> None:
    normalized_scores = normalize._subtract_min_score(scores, offset=offset)
    finite_scores = [s for s in scores if not np.isnan(s) and not np.isinf(s)]
    min_score = min(finite_scores) if len(finite_scores) > 0 else np.nan
    for original_score, normed_score in zip(scores, normalized_scores):
        if np.isnan(original_score):
            assert np.isnan(normed_score)
        elif np.isinf(original_score):
            assert np.isinf(normed_score)
        else:
            assert normed_score == original_score - min_score + offset
