import json
import math
import pickle
from typing import Any, Iterable

import datasets
import numpy as np
import pytest
from datasets import fingerprint

from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.pipes.utils.misc import iter_examples, pack_examples


@pytest.fixture
def corpus(
    seed: int,
    num_kbs: int = 10,
    num_answers_per_kbs: tuple[int, int] = (1, 10),
    num_sections_per_answers: tuple[int, int] = (1, 3),
) -> datasets.Dataset:
    rgn = np.random.RandomState(seed)
    data = []
    answer_id = 0
    section_id = 0
    for kb_id in range(num_kbs):
        kb_size = rgn.randint(*num_answers_per_kbs)
        for a in range(kb_size):
            answer_size = rgn.randint(*num_sections_per_answers)
            for s in range(answer_size):
                section = {"id": section_id, "kb_id": kb_id, "answer_id": answer_id}
                data.append(section)
                section_id += 1
            answer_id += 1

    dataset = _convert_rows_to_dataset(data)
    return dataset


@pytest.fixture
def questions(
    seed: int,
    corpus: datasets.Dataset,
    has_section_prob: float = 0.5,
    num_questions: int = 10,
) -> datasets.Dataset:
    """Generate question by randomly"""
    rgn = np.random.RandomState(seed)
    lookups = pipes.lookup_index._build_lookups(corpus, keys=["id", "kb_id", "answer_id"])
    unique_answer_ids = set(lookups["answer_id"].keys())

    data = []
    for qid in range(num_questions):
        is_section = rgn.random() < has_section_prob
        if is_section:
            row_id = rgn.choice(list(range(len(corpus))))
            section = corpus[int(row_id)]
            row = {
                "question_id": qid,
                "section_id": section["id"],
                "answer_id": section["answer_id"],
                "kb_id": section["kb_id"],
            }
        else:
            answer_id = rgn.choice(list(unique_answer_ids))
            row_ids = lookups["answer_id"][answer_id]
            sections = corpus[row_ids]
            kb_ids = set(sections["kb_id"])
            assert len(kb_ids) == 1
            kb_id = kb_ids.pop()
            row = {
                "question_id": qid,
                "section_id": None,
                "answer_id": answer_id,
                "kb_id": kb_id,
            }
        data.append(row)

    return _convert_rows_to_dataset(data)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_sections", [9, 101])
@pytest.mark.parametrize("out_of_domain_score", [-math.inf, None])
def test_lookup_index(
    corpus: datasets.Dataset,
    questions: datasets.Dataset,
    n_sections: int,
    out_of_domain_score: int,
):
    lookup_pipe = pipes.LookupIndexPipe(corpus=corpus, n_sections=n_sections, out_of_domain_score=out_of_domain_score)
    pipe_hash = fingerprint.Hasher.hash(lookup_pipe)

    # process the questions and check the output
    output = lookup_pipe(questions[:])

    # check the output keys
    assert set(output.keys()) == {
        pipes.LookupIndexPipe._pid_key,
        pipes.LookupIndexPipe._score_key,
        pipes.LookupIndexPipe._label_key,
    }

    # check the output consistency
    label_keys = lookup_pipe._label_keys
    in_domain_keys = lookup_pipe._in_domain_keys
    for i, example in enumerate(iter_examples(output)):
        question = questions[i]
        for result in iter_examples(example):
            # fetch the corresponding section
            result_score = result[pipes.LookupIndexPipe._score_key]
            if result_score not in (lookup_pipe._in_domain_score, lookup_pipe._out_of_domain_score):
                raise ValueError(
                    f"Unexpected score: {result_score}. "
                    f"Expected {lookup_pipe._in_domain_score} or {lookup_pipe._out_of_domain_score}"
                )
            pid = result[pipes.LookupIndexPipe._pid_key]
            section = corpus[pid]

            # check the validity of results labelled as positives
            # positive sections should have the same labels as the question
            # and should be defined as in-domain
            is_labelled_as_positive = result[pipes.LookupIndexPipe._label_key]
            if is_labelled_as_positive:
                assert result[pipes.LookupIndexPipe._score_key] == lookup_pipe._in_domain_score
                for q_label_key, s_label_key in label_keys.items():
                    if question[q_label_key] is None:
                        continue
                    if question[q_label_key] != section[s_label_key]:
                        raise ValueError(
                            f"The labels are inconsistent for keys {q_label_key, s_label_key}! "
                            f"Question={question}, Section={section}"
                        )

                # check that the label is in-domain
                for q_domain_key, s_domain_key in in_domain_keys.items():
                    if question[q_domain_key] is None:
                        continue
                    if question[q_domain_key] != section[s_domain_key]:
                        raise ValueError(f"The domains are inconsistent! Question={question}, Section={section}")

            else:
                # check the validity of results labelled as in-domain negatives
                # in-domain negative should have zero-score
                if lookup_pipe._in_domain_score == lookup_pipe._out_of_domain_score:
                    continue

                is_in_domain = result_score == lookup_pipe._in_domain_score
                if is_in_domain:
                    for q_domain_key, s_domain_key in in_domain_keys.items():
                        if question[q_domain_key] is None:
                            continue
                        if question[q_domain_key] != section[s_domain_key]:
                            raise ValueError(
                                f"The domain labels are inconsistent for keys {q_domain_key, s_domain_key}! "
                                f"Question={question}, Section={section}"
                            )

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
