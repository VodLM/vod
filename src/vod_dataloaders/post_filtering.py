from __future__ import annotations

import abc
import math
from typing import Any, TypeVar

import numpy as np
from raffle_ds_research.tools import dstruct

K = TypeVar("K")


class PostFilter(abc.ABC):
    """A post-filtering method."""

    @abc.abstractmethod
    def __call__(
        self,
        indices: np.ndarray,
        scores: dict[K, np.ndarray],
        *,
        query: dict[str, Any],
    ) -> dict[K, np.ndarray]:
        """Filter the results."""
        raise NotImplementedError()


class EnsureMatch(PostFilter):
    """Ensure that the retrieved sections match the query."""

    __slots__ = ("_features", "_query_key", "_section_key")
    _features: np.ndarray
    _query_key: str
    _section_key: str

    def __init__(
        self,
        *,
        sections: dstruct.SizedDataset[dict[str, Any]],
        query_key: str = "group_hash",
        section_key: str = "group_hash",
    ):
        """Initialize the post-filtering method."""
        self._query_key = query_key
        self._section_key = section_key

        # build the features
        self._features = np.full((len(sections),), -1, dtype=np.int64)
        for i in range(len(sections)):
            section = sections[i]
            self._features[i] = section[self._section_key]

    def __call__(
        self,
        indices: np.ndarray,
        scores: dict[K, np.ndarray],
        *,
        query: dict[str, Any],
    ) -> dict[K, np.ndarray]:
        """Filter the results."""
        batch_features = np.asarray(query[self._query_key], dtype=np.int64)
        results_features = self._features[indices]
        keep_mask = batch_features[:, None] == results_features
        keep_mask |= (keep_mask.sum(axis=1) == 0)[:, None]  # <- don't filter out rows without match
        return {key: np.where(keep_mask, s, -math.inf) for key, s in scores.items()}


def post_filter_factory(
    mode: str = "ensure_match",
    *,
    sections: dstruct.SizedDataset[dict[str, Any]],
    query_key: str = "group_hash",
    section_key: str = "group_hash",
) -> PostFilter:
    """Build a post-filtering method."""
    if mode == "ensure_match":
        return EnsureMatch(sections=sections, query_key=query_key, section_key=section_key)

    raise ValueError(f"Unknown post-filtering mode: {mode}")
