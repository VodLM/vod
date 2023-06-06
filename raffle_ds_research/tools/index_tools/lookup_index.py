from __future__ import annotations

import collections
import dataclasses
import functools
import multiprocessing as mp
import pathlib
import pickle
import random
import sys
from typing import Any, Callable, Iterable, TypeVar

import datasets
import numpy as np
from rich.progress import track

from raffle_ds_research.tools import dstruct, pipes
from raffle_ds_research.tools.index_tools.retrieval_data_type import RetrievalBatch

T = TypeVar("T")


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


@dataclasses.dataclass
class LookupValue:
    """Models a value to index."""

    index: int
    label: int
    group: int


def _build_lookup_table_grouped(values: Iterable[LookupValue]) -> dict[int, dict[int, set[int]]]:
    """Build the lookup table."""
    lookup_table: dict[int, dict[int, set[int]]] = collections.defaultdict(__defaultdict_of_sets)
    for v in values:
        lookup_table[v.group][v.label].add(v.index)

    return lookup_table


def _yield_chunks(values: Iterable[T], chunk_size: int) -> Iterable[list[T]]:
    chunk = []
    for v in values:
        chunk.append(v)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    yield chunk


def _build_part(
    ids: list[int],
    corpus: dstruct.SizedDataset[dict[str, Any]],
    cast_fn: Callable[[int, dict[str, Any]], LookupValue],
) -> dict[int, dict[int, set[int]]]:
    values = (cast_fn(i, corpus[i]) for i in ids)
    return _build_lookup_table_grouped(values)


def _build_lookup_table_grouped_parallel(
    corpus: dstruct.SizedDataset[dict[str, Any]],
    cast_fn: Callable[[int, dict[str, Any]], LookupValue],
    num_proc: int = 4,
    chunk_size: int = 50_000,
) -> dict[int, dict[int, set[int]]]:
    lookup_table: dict[int, dict[int, set[int]]] = collections.defaultdict(__defaultdict_of_sets)

    with mp.Pool(processes=num_proc) as pool:
        for part_table in track(
            pool.imap_unordered(
                functools.partial(_build_part, corpus=corpus, cast_fn=cast_fn),  # type: ignore
                _yield_chunks(range(len(corpus)), chunk_size),
            ),
            total=len(corpus) // chunk_size,
            description="Building lookup table",
        ):
            # update the master lookup table with the partial lookup table
            for group, group_lookup in part_table.items():
                for label, indices in group_lookup.items():
                    lookup_table[group][label].update(indices)

    return lookup_table


def _cast_fn(index: int, eg: dict[str, Any], label_key: str, group_key: str) -> LookupValue:
    return LookupValue(label=eg[label_key], group=eg[group_key], index=index)


class LookupIndexbyGroup:
    """Retrieve sections with matching labels."""

    __slots__ = ["_key", "_group_key", "_lookup_table", "_corpus_fingperint", "_num_proc"]
    _keys: list[str]
    _lookup_table: dict[int, dict[int, set[int]]]
    _corpus_fingperint: str

    def __init__(self, corpus: dstruct.SizedDataset[dict[str, Any]], key: str, group_key: str, num_proc: int = 4):
        self._corpus_fingperint = pipes.fingerprint(corpus)
        self._key = key
        self._group_key = group_key
        self._num_proc = num_proc

        corpus_column_names = corpus[0].keys()
        if not set(corpus_column_names).issuperset([key, group_key]):
            raise ValueError(f"Keys {[key, group_key]} not found in corpus. Found columns: `{corpus_column_names}`")

        # convert the lookup tables to numpy arrays for more efficient processing
        self._lookup_table = _build_lookup_table_grouped_parallel(
            corpus=corpus,
            cast_fn=functools.partial(_cast_fn, label_key=key, group_key=group_key),
            num_proc=self._num_proc,
        )

    def save(self, path: str | pathlib.Path) -> None:
        """Save the index to disk."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "LookupIndexbyGroup":  # noqa: ANN102
        """Load the index from disk."""
        path = pathlib.Path(path)
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    def validate(
        self,
        corpus: dstruct.SizedDataset[dict[str, Any]],
        num_samples: int = 1_000,
    ) -> None:
        """Check if the corpus has changed since the index was created."""
        if pipes.fingerprint(corpus) != self._corpus_fingperint:
            raise ValueError("The corpus has changed since the index was created.")

        # take a few sample and check the lookup consistency
        cast_fn = functools.partial(
            _cast_fn,
            label_key=self._key,
            group_key=self._group_key,
        )
        ids = random.sample(range(len(corpus)), num_samples)
        for i in track(ids, description="Validating lookup table"):
            eg = corpus[i]
            v = cast_fn(i, eg)
            r = self.search([v.label], [v.group])
            if v.index not in r.indices[0]:
                raise ValueError(f"Lookup table is inconsistent. Expected {v.index} in {r.indices[0]}")

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

    @property
    def memsize(self) -> float:
        """Return the memory size of the index in bytes."""
        return _get_object_size(self._lookup_table)


def _get_object_size(obj: object) -> float:
    size = sys.getsizeof(obj)

    if isinstance(obj, collections.abc.Mapping):
        size += sum(_get_object_size(key) + _get_object_size(value) for key, value in obj.items())
    elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str):
        size += sum(_get_object_size(item) for item in obj)

    return size


@datasets.fingerprint.hashregister(LookupIndexbyGroup)
def _hash_grouped_index(hasher: datasets.fingerprint.Hasher, value: LookupIndexbyGroup) -> str:
    data = {s: getattr(value, s, None) for s in value.__slots__}
    data.pop("_lookup_table")
    return hasher.hash(data)


@datasets.fingerprint.hashregister(LookupIndex)
def _hash_index(hasher: datasets.fingerprint.Hasher, value: LookupIndex) -> str:
    data = {s: getattr(value, s, None) for s in value.__slots__ if s not in {"_lookup_table", "_num_proc"}}
    return hasher.hash(data)
