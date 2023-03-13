import json
import pickle
from typing import Any, Iterable

import datasets
import numpy as np
import pytest
from datasets import fingerprint

from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.pipes.lookup_index import _build_lookup_tables
from raffle_ds_research.tools.pipes.utils.misc import pack_examples

_KEYS = ["kb_id", "answer_id", "section_id"]


def _gen_nested_rows(key: str, total: int, data: list[dict] = None) -> Iterable[dict]:
    if data is None:
        for global_id in range(total):
            yield {key: global_id}
    else:
        global_id = 0
        for row in data:
            for local_id in range(total):
                if key in row:
                    raise ValueError(f"Key {key} already exists in row {row}")
                yield {**row, key: global_id}
                global_id += 1


def gen_corpus(
    keys: list[str],
    size_per_step: int = 10,
) -> datasets.Dataset:
    """Generate a corpus using a nested structure.
    E.g., kb_id -> answer_id -> section_id"""
    data = None
    for key in keys:
        data = _gen_nested_rows(key, size_per_step, data)

    dataset = _convert_rows_to_dataset(data)
    return dataset


def _search_by_value(key: str, value: Any, corpus: datasets.Dataset) -> Iterable[tuple[int, dict]]:
    if value is None:
        return []
    for i, row in enumerate(corpus):
        if row[key] == value:
            yield i, row


def gen_questions(
    seed: int,
    corpus: datasets.Dataset,
    last_level_link_prob: float = 1.0,
    num_questions: int = 10,
) -> datasets.Dataset:
    """
    Generate question by randomly linking to the corpus.
    keys are sampled independently, so the nested structure is not used here..
    """
    rgn = np.random.RandomState(seed)
    keys = corpus.column_names
    lookup_tables = _build_lookup_tables(corpus, keys=keys)

    def _fetch_values(match_key, match_value, output_key) -> list[int]:
        potential_rows = {x[output_key] for i, x in _search_by_value(match_key, match_value, corpus)}
        return list(potential_rows)

    data = []
    for _ in range(num_questions):
        row = {}

        # first level
        first_key, *middle_keys, last_key = keys
        domain_keys = list(lookup_tables[first_key].keys())
        row[first_key] = rgn.choice(domain_keys)

        # intermediate levels
        prev_key = key = first_key
        for key in middle_keys:
            linked_key_values = _fetch_values(prev_key, row[prev_key], key)
            key_value = rgn.choice(linked_key_values)
            row[key] = key_value

        # last level
        if rgn.random() <= last_level_link_prob:
            linked_key_values = _fetch_values(key, row[key], last_key)
            row[last_key] = rgn.choice(linked_key_values)
        else:
            row[last_key] = None

        data.append(row)

    return _convert_rows_to_dataset(data)


@pytest.fixture
def corpus(size_per_step: int) -> datasets.Dataset:
    return gen_corpus(keys=_KEYS, size_per_step=size_per_step)


@pytest.fixture
def questions(
    seed: int,
    corpus: datasets.Dataset,
    has_section_prob: float,
    num_questions: int = 10,
) -> datasets.Dataset:
    return gen_questions(
        seed=seed,
        corpus=corpus,
        last_level_link_prob=has_section_prob,
        num_questions=num_questions,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("size_per_step", [3, 10])
@pytest.mark.parametrize("has_section_prob", [1, 0.5])
def test_lookup_index(
    corpus: datasets.Dataset,
    questions: datasets.Dataset,
):
    lookup_pipe = pipes.LookupIndexPipe(corpus=corpus, keys=_KEYS)
    pipe_hash = fingerprint.Hasher.hash(lookup_pipe)

    # process the questions and check the output
    batch = questions[:]
    output = lookup_pipe(batch)

    # check the name of the output keys
    assert set(output.keys()) == set(lookup_pipe.keys + [lookup_pipe._output_idx_name])

    # check the output values
    retrieved_pids = output[lookup_pipe._output_idx_name]
    for i, pid in enumerate(retrieved_pids):
        for key in _KEYS:
            # search all matches in the corpus (reference data, expensive to compute)
            eg_value = batch[key][i]
            matched_pids = set(i for i, x in _search_by_value(key, eg_value, corpus))

            # retrieve the matched pids in the output of the pipe
            binary_labels = output[key][i]
            found_pids = {int(pid) for pid, label in zip(retrieved_pids[i], binary_labels) if (label > 0 and pid >= 0)}

            # check that the pids are the same
            if matched_pids != found_pids:
                raise RuntimeError(f"Found pids {found_pids} do not match expected pids {matched_pids}!")

    # test that the hash of the pipe is deterministic (it should not change after being used)
    new_pipe_hash = fingerprint.Hasher.hash(lookup_pipe)
    if pipe_hash != new_pipe_hash:
        raise RuntimeError("Pipe hash changed after being used!")

    # test that the pipe can be deterministically serialized
    deserialized_pipe = pickle.loads(pickle.dumps(lookup_pipe))
    deserialized_pipe_hash = fingerprint.Hasher.hash(deserialized_pipe)
    if pipe_hash != deserialized_pipe_hash:
        raise RuntimeError("Pipe hash changed after being serialized!")


def _hash_json_dump(data):
    return datasets.fingerprint.Hasher.hash(json.dumps(data))


def _convert_rows_to_dataset(rows: Iterable[dict[str, Any]]) -> datasets.Dataset:
    dataset = pack_examples(rows)
    return datasets.Dataset.from_dict(dataset)
