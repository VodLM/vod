from __future__ import annotations

import copy
import dataclasses
from typing import Optional

import numpy as np
import rich

from raffle_ds_research.core.mechanics.utils import fill_nans_with_min, numpy_gumbel_like, numpy_log_softmax
from raffle_ds_research.tools import c_tools, index_tools


@dataclasses.dataclass(frozen=True)
class SampledSections:
    """Holds the sampled sections (ids, scores and labels) for a batch of data."""

    indices: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    other_scores: Optional[dict[str, np.ndarray]] = None


def sample_sections(
    *,
    positives: index_tools.RetrievalBatch,
    candidates: index_tools.RetrievalBatch,
    n_sections: Optional[int],
    max_pos_sections: Optional[int],
    do_sample: bool = False,
    other_scores: Optional[dict[str, np.ndarray]] = None,
    lookup_positive_scores: bool = True,
) -> SampledSections:
    """Sample the positive and negative sections.

    This function uses the Gumbel-Max trick to sample from the corresponding distributions.
    Gumbel-Max: https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/.
    """
    if max_pos_sections is None:
        max_pos_sections = positives.indices.shape[-1]
    if n_sections is None:
        n_sections = candidates.indices.shape[-1] + max_pos_sections

    positives = copy.copy(positives)
    if lookup_positive_scores:
        # set the positive scores to be the scores from the pool of candidates
        positives.scores = c_tools.gather_by_index(
            queries=positives.indices,
            keys=candidates.indices,
            values=candidates.scores,
        )
        # replace NaNs with the minimum value along each dimension (each question)
        positives.scores = fill_nans_with_min(offset_min_value=-1, values=positives.scores)
        positives.scores = np.where(positives.indices < 0, -np.inf, positives.scores)
    else:
        positives.scores = np.where(np.isnan(positives.scores), 0, positives.scores)
        positives.scores = np.where(positives.indices < 0, -np.inf, positives.scores)

    # gather the positive sections and apply perturbations
    positive_logits = numpy_log_softmax(positives.scores)
    if do_sample:
        positive_logits += numpy_gumbel_like(positive_logits)

    # gather the negative sections and apply perturbations
    negative_logits = numpy_log_softmax(candidates.scores)
    if do_sample:
        negative_logits += numpy_gumbel_like(negative_logits)

    # concat the positive and negative sections
    concatenated = c_tools.concat_search_results(
        a_indices=positives.indices,
        a_scores=positive_logits,
        b_indices=candidates.indices,
        b_scores=negative_logits,
        max_a=max_pos_sections,
        total=n_sections,
    )

    # set the labels to be `1` for the positive sections (revert label ordering)
    concatenated.labels = np.where(concatenated.labels == 0, 1, 0)

    # fetch the scores from the pool of negatives
    scores = c_tools.gather_by_index(
        queries=concatenated.indices,
        keys=candidates.indices,
        values=candidates.scores,
    )

    # also fetch the `other` scores (e.g., bm25, faiss) for tracking purposes
    if other_scores is not None:
        other_scores = {
            k: c_tools.gather_by_index(
                queries=concatenated.indices,
                keys=candidates.indices,
                values=v,
            )
            for k, v in other_scores.items()
        }

    output = SampledSections(
        indices=concatenated.indices,
        scores=scores,
        labels=concatenated.labels,
        other_scores=other_scores,
    )

    if (concatenated.labels.sum(axis=1) == 0).any():
        rich.print(
            {
                "positive_indices": positives.indices,
                "positive_logits": positive_logits,
                "negative_indices": candidates.indices,
                "negative_logits": negative_logits,
            }
        )
        rich.print(output)
        raise ValueError("No positive sections were sampled.")

    return output
