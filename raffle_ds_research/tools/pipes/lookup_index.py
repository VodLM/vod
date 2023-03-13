from __future__ import annotations

from collections import defaultdict, OrderedDict
from copy import copy
from functools import partial
from typing import Any, Iterable, Optional

import datasets
import numpy as np
from typing_extensions import TypeAlias

from raffle_ds_research.tools import c_tools
from raffle_ds_research.tools.dataset_builder import DatasetProtocol
from raffle_ds_research.tools.pipes.utils.misc import pad_list, iter_examples

LookupTables: TypeAlias = OrderedDict[str, dict[int, set[int]]]


def _pad_pids(pids: list[int], candidate_pids: list[int] | set[int], **kwargs):
    pids = pad_list(pids, fill_values=set(candidate_pids) - set(pids), **kwargs)
    return pids


def _get_defaultdict(cls: type):
    return defaultdict(cls)


def _build_lookup_tables(corpus: DatasetProtocol, keys: Iterable[str]) -> LookupTables:
    """Build the lookup tables"""
    keys = list(keys)
    lookups = defaultdict(partial(_get_defaultdict, set))
    for i, row in enumerate(corpus):
        for key in keys:
            row_value = row[key]
            lookups[key][row_value].add(i)

    return OrderedDict([(k, lookups[k]) for k in keys])


class LookupIndexPipe(object):
    """This Pipe allows retrieving sections with matching key values."""

    __slots__ = [
        "_keys",
        "_corpus",
        "_lookup_tables",
        "_key_maps",
    ]
    _output_idx_name: str = "local_idx"
    _keys: list[str]
    _lookup_tables: dict[str, np.ndarray]
    _key_maps: dict[str, [int, int]]
    _corpus: datasets.Dataset

    def __init__(self, corpus: datasets.Dataset, keys: list[str] = None):
        self._keys = keys
        if not set(self._keys).issubset(corpus.column_names):
            raise ValueError(f"Corpus must have columns: `{self._keys}`")
        self._corpus = corpus

        # convert the lookup tables to numpy arrays for more efficient processing
        py_lookup_tables = _build_lookup_tables(corpus, keys=self._keys)
        self._lookup_tables, self._key_maps = {}, {}
        for key, lookup in py_lookup_tables.items():
            nwors = len(lookup)
            ncols = max(len(v) for v in lookup.values())
            key_lookup = np.full((nwors, ncols), -1, dtype=np.int64)
            key_map = {}
            for idx, (k, v) in enumerate(lookup.items()):
                key_map[k] = idx
                key_lookup[idx, : len(v)] = list(v)

            self._lookup_tables[key] = key_lookup
            self._key_maps[key] = key_map

    @property
    def keys(self) -> list[str]:
        return copy(self._keys)

    def __call__(
        self,
        batch: dict[str, Any],
        idx: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        # retrieve the pids for each query and key
        # todo: cythonize this for loop.
        nrows = len(batch[self._keys[0]])
        ncols = sum(v.shape[1] for v in self._lookup_tables.values())
        batch_pids = np.full((nrows, ncols), fill_value=-1, dtype=np.int64)
        batch_labels = np.full((nrows, ncols), fill_value=-1, dtype=np.int64)
        for i, eg in enumerate(iter_examples(batch, keys=self._keys)):
            cursor = 0
            for j, key in enumerate(self._keys):
                key_map = self._key_maps[key]
                key_lookup = self._lookup_tables[key]
                eg_key = eg[key]
                if eg_key is None:
                    key_pids = np.full_like(key_lookup[0], fill_value=-1, dtype=np.int64)
                else:
                    key_pids = key_lookup[key_map[eg_key]]
                a = cursor
                cursor += len(key_pids)
                batch_pids[i, a:cursor] = key_pids
                batch_labels[i, a:cursor] = j

        # find the unique pids and their labels
        upids, ulabels = c_tools.unique_by_label(batch_pids, batch_labels, n_labels=len(self._keys))

        return {
            self._output_idx_name: upids,
            **{key: ulabels[..., i] for i, key in enumerate(self._keys)},
        }


@datasets.fingerprint.hashregister(LookupIndexPipe)
def _hash_index(hasher, value: LookupIndexPipe):
    data = {s: getattr(value, s, None) for s in value.__slots__}
    data.pop("_lookup_tables")
    if isinstance(value._corpus, datasets.Dataset):
        data["_corpus"] = value._corpus._fingerprint
    h = hasher.hash_bytes(data)
    return h
