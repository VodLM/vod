from __future__ import annotations

import collections
from typing import Iterable

import datasets
import numpy as np

from raffle_ds_research.tools.index_tools.retrieval_data_type import RetrievalBatch


def _build_lookup_table(labels: Iterable[int]) -> dict[int, set[int]]:
    """Build the lookup table"""
    lookup_table = collections.defaultdict(set)
    for i, lbl in enumerate(labels):
        lookup_table[lbl].add(i)

    return lookup_table


class LookupIndex(object):
    """Retrieve sections with matching labels."""

    __slots__ = [
        "_key",
        "_lookup_table",
        "_corpus_fingperint",
    ]
    _keys: str
    _lookup_table: dict[int, set[int]]
    _corpus_fingperint: str

    def __init__(self, corpus: datasets.Dataset, key: str):
        self._corpus_fingperint = corpus._fingerprint
        self._key = key
        if self._key not in corpus.column_names:
            raise ValueError(f"Key {self._key} not found in corpus. Found columns: `{corpus.column_names}`")

        # convert the lookup tables to numpy arrays for more efficient processing
        labels = (eg[self._key] for eg in corpus)
        self._lookup_table = _build_lookup_table(labels)

    def search(self, queries: list[int] | list[list[int]]) -> RetrievalBatch[np.ndarray]:
        """Search for the given key in the lookup table"""

        indices, scores = [], []
        max_n = 0
        for query in queries:
            if isinstance(query, int):
                query = [query]
            q_indices = list(set.union(*[self._lookup_table[q] for q in query if q >= 0]))
            max_n = max(max_n, len(q_indices))
            indices.append(np.array(q_indices))
            scores.append(np.zeros(len(q_indices), dtype=np.float32))

        indices = np.stack([np.pad(i, (0, max_n - len(i)), constant_values=-1) for i in indices])
        scores = np.stack([np.pad(s, (0, max_n - len(s)), constant_values=-np.inf) for s in scores])

        return RetrievalBatch(indices=indices, scores=scores)


@datasets.fingerprint.hashregister(LookupIndex)
def _hash_index(hasher, value: LookupIndex):
    data = {s: getattr(value, s, None) for s in value.__slots__}
    data.pop("_lookup_table")
    h = hasher.hash_bytes(data)
    return h
