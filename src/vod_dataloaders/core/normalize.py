import numpy as np
import numpy.typing as npt
import vod_types as vt


def normalize_search_scores_(search_results: dict[str, vt.RetrievalBatch], offset: float = 0.0) -> None:
    """Subtract the minimum score from all score to allow consistent aggregation.

    NOTE: this method is inplace.
    """
    for key, result in search_results.items():
        if result.scores.size == 0:
            continue
        search_results[key].scores = _subtract_min_score(result.scores, offset=offset)


def _subtract_min_score(scores: npt.NDArray, offset: float) -> npt.NDArray:
    non_nan_scores = np.where(np.isinf(scores) | np.isnan(scores), np.inf, scores)
    min_score = np.amin(non_nan_scores, axis=1, keepdims=True)
    return scores - min_score + offset
