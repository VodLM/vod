from __future__ import annotations

import json
from collections import defaultdict
from functools import partial
from typing import Optional, Type

import numpy as np
from datasets.fingerprint import hashregister
from pydantic import PrivateAttr

from .index import Index, SupervisedIndexInput, SupervisedIndexOuput
from ..utils.misc import pad_list


class LookupIndex(Index[SupervisedIndexInput, SupervisedIndexOuput, None]):
    _input_model: Optional[Type[SupervisedIndexInput]] = PrivateAttr(SupervisedIndexInput)
    _output_model: Optional[Type[SupervisedIndexOuput]] = PrivateAttr(SupervisedIndexOuput)
    _required_sections_columns: set[int] = PrivateAttr({"id", "answer_id"})
    _section_lookup: dict[int, int] = PrivateAttr(None)
    _answer_lookup: dict[int, list[int]] = PrivateAttr(None)

    def _build_index(self):
        """Build the index (e.g., faiss)"""
        self._section_lookup = {s["id"]: i for i, s in enumerate(self._sections)}
        self._answer_lookup = defaultdict(list)
        for i, s in enumerate(self._sections):
            self._answer_lookup[s["answer_id"]].append(i)

    def _process_batch(self, batch: dict, idx: Optional[list[int]] = None, **kwargs) -> dict:
        batch_pids = []
        for sid, aid in zip(batch["section_id"], batch["answer_id"]):
            # lookup the pids
            if sid is not None:
                pids = [self._section_lookup[sid]]
            else:
                pids = self._answer_lookup[aid]

            # resample if we have too many
            if self.max_top_k is not None and len(pids) > self.max_top_k:
                pids = np.random.choice(pids, self.max_top_k, replace=False)

            batch_pids.append(pids)

        # generate the scores
        batch_scores = [[0.0] * len(pids) for pids in batch_pids]
        batch_labels = [[True] * len(pids) for pids in batch_pids]
        # pad the scores and pids
        if self.padding:
            all_pids = set(range(len(self._sections)))
            pid_pad_fn = partial(self.pad_pids, length=self.max_top_k, candidate_pids=all_pids)
            score_pad_fn = partial(pad_list, length=self.max_top_k, fill_value=self.padding_score)
            label_pad_fn = partial(
                pad_list,
                length=self.max_top_k,
                fill_value=False,
            )
            batch_pids = list(map(pid_pad_fn, batch_pids))
            batch_scores = list(map(score_pad_fn, batch_scores))
            batch_labels = list(map(label_pad_fn, batch_labels))

        output = {
            self.pid_key: batch_pids,
            self.score_key: batch_scores,
            self.label_key: batch_labels,
        }

        return output

    @staticmethod
    def pad_pids(pids: list[int], candidate_pids: list[int] | set[int], **kwargs):
        pids = pad_list(pids, fill_values=candidate_pids - set(pids), **kwargs)
        return pids


@hashregister(Index, LookupIndex)
def _hash_index(hasher, value: Index):
    data = value.dict()
    data["_sections"] = value._sections._fingerprint
    h = hasher.hash_bytes(json.dumps(data, sort_keys=True).encode("utf-8"))
    return h
