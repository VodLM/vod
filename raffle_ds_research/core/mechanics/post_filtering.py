from __future__ import annotations

import abc
import math
from typing import Any

import datasets
import numpy as np

from raffle_ds_research.tools import dstruct, index_tools, pipes


class PostFilter(abc.ABC):
    """A post-filtering method."""

    @abc.abstractmethod
    def __call__(
        self,
        results: index_tools.RetrievalBatch,
        *,
        query: dict[str, Any],
    ) -> index_tools.RetrievalBatch:
        """Filter the results."""
        raise NotImplementedError()


def _ensure_match(
    *,
    batch: dict[str, Any],
    candidate_samples: index_tools.RetrievalBatch,
    features: dstruct.SizedDataset[dict[str, Any]],
    features_keys: list[str],
) -> index_tools.RetrievalBatch:
    """Filter the candidates sections based on the `config.ensure_match` keys."""
    batch_features = {key: np.asarray(batch[key]) for key in features_keys}
    section_indices = candidate_samples.indices.flatten().tolist()
    section_features = features[section_indices]

    keep_mask = None
    for key, batch_features_key in batch_features.items():
        section_features_key = section_features[key]
        section_features_key = np.asarray(section_features_key).reshape(batch_features_key.shape[0], -1)
        keep_mask_key = batch_features_key[:, None] == section_features_key
        keep_mask = keep_mask_key if keep_mask is None else keep_mask & keep_mask_key
    if keep_mask is None:
        raise ValueError("No features to match")

    candidate_samples.scores = np.where(keep_mask, candidate_samples.scores, -math.inf)
    return candidate_samples


def _gather_features(
    sections: dstruct.SizedDataset[dict[str, Any]],
    features: list[str],
) -> dstruct.SizedDataset[dict[str, Any]]:
    """Gather the selected features from the sections."""
    if isinstance(sections, datasets.Dataset):
        return pipes.misc.keep_only_columns(sections, features)  # type: ignore

    if isinstance(sections, dstruct.ConcatenatedSizedDataset):
        return dstruct.ConcatenatedSizedDataset(parts=[_gather_features(p, features) for p in sections.parts])

    return sections
