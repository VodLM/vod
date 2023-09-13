import math

import numpy as np
from vod_dataloaders.tools import fast

from src import vod_search


def flatten_sections(
    samples: vod_search.RetrievalBatch,
    search_results: vod_search.RetrievalBatch,
    raw_scores: dict[str, np.ndarray],
    world_size: int,
    padding: bool = True,
) -> tuple[vod_search.RetrievalBatch, dict[str, np.ndarray]]:
    """Merge all sections (positive and negative) as a flat batch."""
    unique_indices = np.unique(samples.indices)
    if padding:
        # pad the unique indices with random indices to a fixed size
        n_full = math.prod(samples.indices.shape)
        n_pad = n_full - unique_indices.shape[0]
        # We sample sections iid so there might be some collisions
        #   but in practice this is sufficiently unlikely,
        #   so it's not worth the extra computation to check.
        random_indices = np.random.randint(0, world_size, size=n_pad)
        unique_indices = np.concatenate([unique_indices, random_indices])

    # Repeat the unique indices for each section
    unique_indices_: np.ndarray = unique_indices[None, :].repeat(samples.indices.shape[0], axis=0)

    # Gather the scores from the `candidates` batch, set the NaNs to the minimum score
    scores = fast.gather_values_by_indices(unique_indices_, search_results.indices, search_results.scores)

    # Gather the labels from the `positives` batch, set NaNs to negatives
    if search_results.labels is None:
        raise ValueError("The `search_results` must have labels.")
    labels = fast.gather_values_by_indices(unique_indices_, search_results.indices, search_results.labels)
    labels[np.isnan(labels)] = -1

    # Other scores (client scores)
    flat_raw_scores = {}
    for key in raw_scores:
        flat_raw_scores[key] = fast.gather_values_by_indices(unique_indices_, search_results.indices, raw_scores[key])

    return (
        vod_search.RetrievalBatch(
            indices=unique_indices,
            scores=scores,
            labels=labels,
            allow_unsafe=True,
        ),
        flat_raw_scores,
    )
