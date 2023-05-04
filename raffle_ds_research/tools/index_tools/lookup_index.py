from __future__ import annotations

import collections
from typing import Any, Iterable

import datasets
import numpy as np

from raffle_ds_research.tools import dstruct, pipes
from raffle_ds_research.tools.index_tools.retrieval_data_type import RetrievalBatch


def _build_lookup_table(labels: Iterable[int]) -> dict[int, set[int]]:
    """Build the lookup table."""
    lookup_table = collections.defaultdict(set)
    for i, lbl in enumerate(labels):
        lookup_table[lbl].add(i)

    return lookup_table


class LookupIndex:
    """Retrieve sections with matching labels."""

    __slots__ = [
        "_key",
        "_lookup_table",
        "_corpus_fingperint",
    ]
    _keys: str
    _lookup_table: dict[int, set[int]]
    _corpus_fingperint: str

    def __init__(self, corpus: dstruct.SizedDataset[dict[str, Any]], key: str):
        self._corpus_fingperint = pipes.fingerprint(corpus)
        self._key = key
        corpus_column_names = corpus[0].keys()
        if self._key not in corpus_column_names:
            raise ValueError(f"Key {self._key} not found in corpus. Found columns: `{corpus_column_names}`")

        # convert the lookup tables to numpy arrays for more efficient processing
        # todo: adapt this line for the new indexable object
        labels = (eg[self._key] for eg in iter(corpus))
        self._lookup_table = _build_lookup_table(labels)

    def search(self, queries: list[int] | list[list[int]]) -> RetrievalBatch[np.ndarray]:
        """Search for the given key in the lookup table."""
        indices, scores = [], []
        max_n = 0
        for query in queries:
            if isinstance(query, int):
                query = [query]  # noqa: PLW2901
            q_indices = list(set.union(*[self._lookup_table[q] for q in query if q >= 0]))
            max_n = max(max_n, len(q_indices))
            indices.append(np.array(q_indices))
            scores.append(np.zeros(len(q_indices), dtype=np.float32))

        indices = np.stack([np.pad(i, (0, max_n - len(i)), constant_values=-1) for i in indices])
        scores = np.stack([np.pad(s, (0, max_n - len(s)), constant_values=-np.inf) for s in scores])

        return RetrievalBatch(indices=indices, scores=scores)


def __defaultdict_of_sets() -> dict[Any, set[Any]]:
    return collections.defaultdict(set)


def _build_lookup_table_kb(labels: Iterable[int], kb_labels: Iterable[int]) -> dict[int, dict[int, set[int]]]:
    """Build the lookup table."""
    lookup_table: dict[int, dict[int, set[int]]] = collections.defaultdict(__defaultdict_of_sets)
    for i, (lbl, kb_lbl) in enumerate(zip(labels, kb_labels)):
        lookup_table[kb_lbl][lbl].add(i)

    return lookup_table


class LookupIndexbyGroup:
    """Retrieve sections with matching labels."""

    __slots__ = [
        "_key",
        "_group_key",
        "_lookup_table",
        "_corpus_fingperint",
    ]
    _keys: list[str]
    _lookup_table: dict[int, dict[int, set[int]]]
    _corpus_fingperint: str

    def __init__(self, corpus: dstruct.SizedDataset[dict[str, Any]], key: str, group_key: str):
        self._corpus_fingperint = pipes.fingerprint(corpus)
        self._key = key
        self._group_key = group_key

        corpus_column_names = corpus[0].keys()
        if not set(corpus_column_names).issuperset([key, group_key]):
            raise ValueError(f"Keys {[key, group_key]} not found in corpus. Found columns: `{corpus_column_names}`")

        # convert the lookup tables to numpy arrays for more efficient processing
        self._lookup_table = _build_lookup_table_kb(
            *zip(*[(eg[self._key], eg[self._group_key]) for eg in iter(corpus)])
        )

    def search(self, labels: list[list[int]] | list[int], groups: list[int]) -> RetrievalBatch[np.ndarray]:
        """Search for the given key in the lookup table."""
        indices, scores = [], []
        max_n = 0
        for query, group in zip(labels, groups):
            if isinstance(query, int):
                query = [query]  # noqa: PLW2901
            lookup_table = self._lookup_table[group]
            q_indices = list(set.union(*[lookup_table[q] for q in query if q >= 0]))
            max_n = max(max_n, len(q_indices))
            indices.append(np.array(q_indices))
            scores.append(np.zeros(len(q_indices), dtype=np.float32))

        indices = np.stack([np.pad(i, (0, max_n - len(i)), constant_values=-1) for i in indices])
        scores = np.stack([np.pad(s, (0, max_n - len(s)), constant_values=-np.inf) for s in scores])

        return RetrievalBatch(indices=indices, scores=scores)


@datasets.fingerprint.hashregister(LookupIndexbyGroup)
def _hash_grouped_index(hasher: datasets.fingerprint.Hasher, value: LookupIndexbyGroup) -> str:
    data = {s: getattr(value, s, None) for s in value.__slots__}
    data.pop("_lookup_table")
    return hasher.hash(data)


@datasets.fingerprint.hashregister(LookupIndex)
def _hash_index(hasher: datasets.fingerprint.Hasher, value: LookupIndex) -> str:
    data = {s: getattr(value, s, None) for s in value.__slots__}
    data.pop("_lookup_table")
    return hasher.hash(data)
