import math

import numpy as np
import vod_types as vt
from vod_dataloaders.core import numpy_ops

from .sample import PrioritySampledSections


def flatten_samples(samples: PrioritySampledSections, padding: bool = True) -> PrioritySampledSections:
    """Merge all sections (positive and negative) as a flat batch."""
    indices = samples.batch.indices
    unique_indices = np.unique(indices)
    if padding:
        # pad the unique indices with -1
        n_full = math.prod(samples.batch.indices.shape)
        n_pad = n_full - unique_indices.shape[0]
        random_indices = np.ones((n_pad,), dtype=np.int64)
        unique_indices = np.concatenate([unique_indices, random_indices])

    # Repeat the unique indices for each section
    unique_indices_: np.ndarray = unique_indices[None, :].repeat(samples.batch.shape[0], axis=0)

    # Gather the scores from the `candidates` batch, set the NaNs to the minimum score
    scores = numpy_ops.gather_values_by_indices(unique_indices_, indices, samples.batch.scores)

    # Gather the labels from the `positives` batch, set NaNs to negatives
    if samples.batch.labels is None:
        raise ValueError("The `search_results` must have labels.")
    labels = numpy_ops.gather_values_by_indices(unique_indices_, indices, samples.batch.labels, fill_value=0)

    # Gather the `log_weights`
    log_weights = numpy_ops.gather_values_by_indices(unique_indices_, indices, samples.log_weights)

    # Other scores (client scores)
    flat_raw_scores = {}
    for key in samples.raw_scores:
        flat_raw_scores[key] = numpy_ops.gather_values_by_indices(unique_indices_, indices, samples.raw_scores[key])

    return PrioritySampledSections(
        batch=vt.RetrievalBatch(
            indices=unique_indices,
            scores=scores,
            labels=labels,
            allow_unsafe=True,
        ),
        raw_scores=flat_raw_scores,
        log_weights=log_weights,
    )
