import numpy as np
import vod_search
from vod_dataloaders.tools import fast


def sample_sections(
    *,
    search_results: vod_search.RetrievalBatch,
    raw_scores: dict[str, np.ndarray],
    n_sections: int,
    max_pos_sections: None | int,
    temperature: float = 0,
    max_support_size: None | int = None,
) -> tuple[vod_search.RetrievalBatch, dict[str, np.ndarray]]:
    """Sample the positive and negative sections."""
    samples = fast.sample(
        search_results=search_results,
        total=n_sections,
        n_positives=max_pos_sections,
        temperature=temperature,
        max_support_size=max_support_size,
    )

    # Sample the `raw_scores`
    sampled_raw_scores = {}
    for key, scores in raw_scores.items():
        sampled_raw_scores[key] = fast.gather_values_by_indices(samples.indices, search_results.indices, scores)

    # Set -inf to the mask section (index -1)
    is_masked = samples.indices < 0
    samples.scores.setflags(write=True)
    samples.scores[is_masked] = -np.inf
    for scores in sampled_raw_scores.values():
        scores.setflags(write=True)
        scores[is_masked] = -np.inf

    return samples, sampled_raw_scores
