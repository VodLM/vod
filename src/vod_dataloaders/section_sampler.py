from __future__ import annotations

from typing import Optional

import numpy as np

from raffle_ds_research.core.mechanics import fast
from raffle_ds_research.tools import index_tools


def sample_sections(
    *,
    search_results: index_tools.RetrievalBatch,
    raw_scores: dict[str, np.ndarray],
    n_sections: int,
    max_pos_sections: Optional[int],
    temperature: float = 0,
    max_support_size: Optional[int] = None,
) -> tuple[index_tools.RetrievalBatch, dict[str, np.ndarray]]:
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
