from __future__ import annotations

import math
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Iterable, Optional

import datasets
import numpy as np
from typing_extensions import TypeAlias

from raffle_ds_research.tools.dataset_builder import DatasetProtocol
from raffle_ds_research.tools.pipes.utils.misc import iter_examples, pack_examples, pad_list


def _pad_pids(pids: list[int], candidate_pids: list[int] | set[int], **kwargs):
    pids = pad_list(pids, fill_values=set(candidate_pids) - set(pids), **kwargs)
    return pids


LookupTables: TypeAlias = dict[str, dict[int, set[int]]]


def _get_defaultdict(cls: type):
    return defaultdict(cls)


def _build_lookups(corpus: DatasetProtocol, keys: Iterable[str]) -> LookupTables:
    """Build the lookup tables"""
    keys = list(keys)

    lookups = defaultdict(partial(_get_defaultdict, set))
    for i, row in enumerate(corpus):
        for key in keys:
            row_value = row[key]
            lookups[key][row_value].add(i)

    return lookups


class LookupIndexPipe(object):
    """This Pipe allows retrieving sections from a corpus based on the following strategy:
    1. Retrieve the sections with a match on `label_keys`
        Each returned section is of the form (positive):
        `{"section.pids": ..., "section.label": True, "section.score`: 0.0}`
    2. Retrieve the sections with a match on `in_domain_keys`
        Each returned section is of the form (negative):
        `{"section.pids": ..., "section.label": False, "section.score`: 0.0}`
    3. Fill the remaining sections with other sections from the corpus
        Each returned section is of the form (padding):
        `{"section.pids": ..., "section.label": False, "section.score`: -math.inf}`
    """

    __slots__ = [
        "_corpus",
        "_lookups",
        "_answer_lookup",
        "_n_sections",
        "_in_domain_score",
        "_out_of_domain_score",
        "_all_pids",
        "_label_keys",
        "_in_domain_keys",
    ]
    _label_keys: dict[str, str]
    _in_domain_keys: Optional[dict[str, str]]
    _pid_key: str = "section.pid"
    _score_key: str = "section.score"
    _label_key: str = "section.label"
    _lookups: dict[str, dict[int, set[int]]]
    _answer_lookup: dict[int, list[int]]
    _corpus: datasets.Dataset
    _all_pids: set[int]
    _n_sections: int
    _in_domain_score: float
    _out_of_domain_score: float

    def __init__(
        self,
        *,
        corpus: datasets.Dataset,
        n_sections: int = 100,
        in_domain_score: float = 0.0,
        out_of_domain_score: float = -math.inf,
        label_keys: dict[str, str] = None,
        in_domain_keys: dict[str, str] = None,
    ):
        if label_keys is None:
            label_keys = {
                "section_id": "id",
                "answer_id": "answer_id",
            }
        if len(label_keys) == 0:
            raise ValueError("Must have at least one positive label key. Found zero.")
        self._label_keys = label_keys

        if in_domain_keys is None:
            in_domain_keys = {
                "kb_id": "kb_id",
            }
        self._in_domain_keys = in_domain_keys

        if not set(self._required_corpus_columns).issubset(corpus.column_names):
            raise ValueError(f"Corpus must have columns: `{self._required_corpus_columns}`")

        self._n_sections = n_sections
        self._in_domain_score = in_domain_score
        self._out_of_domain_score = out_of_domain_score
        self._corpus = corpus
        self._lookups = _build_lookups(corpus, keys=self._required_corpus_columns)
        self._all_pids = set(range(len(self._corpus)))

    @property
    def _required_corpus_columns(self) -> set[str]:
        return set(self._label_keys.values()).union(set(self._in_domain_keys.values()))

    @property
    def _required_batch_keys(self) -> set[str]:
        return set(self._label_keys.keys()).union(set(self._in_domain_keys.keys()))

    def __call__(
        self,
        batch: dict[str, Any],
        idx: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> dict[str, list[bool | float | int]]:
        examples = iter_examples(batch, keys=self._required_batch_keys)
        egs = [self._process_example(eg) for eg in examples]
        return pack_examples(egs)

    def _process_example(self, eg: dict[str, Any]) -> dict[str, float | int | bool]:
        eg_pids = defaultdict(list)
        eg_labels = defaultdict(list)
        eg_scores = defaultdict(list)

        def n_pids():
            return sum(len(pids) for pids in eg_pids.values())

        # sample the positive pids
        split = "positive"
        positive_pids = self._gather_pids(
            eg,
            self._label_keys,
            reduce_op=set.intersection,
        )
        positive_pids = self._resample_pids(positive_pids, n=self._n_sections)
        eg_pids[split] = list(positive_pids)
        eg_labels[split] = [True] * len(positive_pids)
        eg_scores[split] = [self._in_domain_score] * len(positive_pids)

        # complete the list of pids with negative pids and the rest
        if len(eg_pids) < self._n_sections:
            if self._in_domain_keys is not None:
                split = "negative"
                neg_eg_pids = self._gather_pids(
                    eg,
                    self._in_domain_keys,
                    exclude=positive_pids,
                    reduce_op=set.intersection,
                )
                neg_eg_pids = self._resample_pids(neg_eg_pids, n=self._n_sections - n_pids())
                eg_pids[split] = list(neg_eg_pids)
                eg_labels[split] = [False] * len(neg_eg_pids)
                eg_scores[split] = [self._in_domain_score] * len(neg_eg_pids)

            # sample with the rest
            if len(eg_pids) < self._n_sections:
                split = "padding"
                remaining_pids = self._all_pids - set.intersection(*(set(x) for x in eg_pids.values()))
                padding_pids = self._resample_pids(remaining_pids, n=self._n_sections - n_pids())
                eg_pids[split] = list(padding_pids)
                eg_labels[split] = [False] * len(padding_pids)
                eg_scores[split] = [self._out_of_domain_score] * len(padding_pids)

        # safety first
        if set.intersection(*(set(x) for x in eg_pids.values())):
            raise ValueError("Duplicate pids found in example")

        # pack the examples
        split_keys = list(eg_pids.keys())
        return {
            self._pid_key: sum([eg_pids[k] for k in split_keys], []),
            self._label_key: sum([eg_labels[k] for k in split_keys], []),
            self._score_key: sum([eg_scores[k] for k in split_keys], []),
        }

    def _gather_pids(
        self,
        eg: dict[str, Any],
        key_map: dict[str, str],
        exclude: Optional[set[int]] = None,
        reduce_op: Callable[[set, set], set] = set.intersection,
    ) -> set[int]:
        pids = None
        for batch_key, corpus_key in key_map.items():
            id_for_key = eg[batch_key]
            if id_for_key is None:
                continue

            # fetch the pids for the key
            lookup_for_key = self._lookups[corpus_key]
            pids_for_key = lookup_for_key[id_for_key]
            if exclude is not None:
                pids_for_key = pids_for_key - exclude

            if pids is None:
                pids = pids_for_key
            else:
                pids = reduce_op(pids, pids_for_key)

        return pids or set()

    @staticmethod
    def _resample_pids(pids: set[int], n: int) -> set[int]:
        if len(pids) > n:
            pids = set(np.random.choice(list(pids), n, replace=False).tolist())
        return pids


@datasets.fingerprint.hashregister(LookupIndexPipe)
def _hash_index(hasher, value: LookupIndexPipe):
    data = {s: getattr(value, s, None) for s in value.__slots__}
    data.pop("_lookups")
    if isinstance(value._corpus, datasets.Dataset):
        data["_corpus"] = value._corpus._fingerprint
    h = hasher.hash_bytes(data)
    return h
