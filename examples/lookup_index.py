import json
import pickle
from typing import Any, Iterable

import datasets
import numpy as np
import rich

from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.pipes.lookup_index import _build_lookups


def hash_json_like(data):
    return datasets.fingerprint.Hasher.hash(json.dumps(data))


def gen_corpus(
    num_kbs: int = 10,
    num_answers_per_kbs: tuple[int, int] = (1, 10),
    num_sections_per_answers: tuple[int, int] = (1, 3),
    rgn: np.random.RandomState = np.random.RandomState(0),
) -> datasets.Dataset:
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

    dataset = convert_rows_to_dataset(data)
    return dataset


def convert_rows_to_dataset(rows: Iterable[dict[str, Any]]) -> datasets.Dataset:
    first_row, *other_rows = rows
    keys = set(first_row.keys())
    dataset = {key: [] for key in keys}
    for row in (first_row, *other_rows):
        for key in keys:
            dataset[key].append(row[key])

    return datasets.Dataset.from_dict(dataset)


def _gen_questions(
    corpus: datasets.Dataset,
    has_section_prob: float = 0.5,
    num_questions: int = 100,
    rgn: np.random.RandomState = np.random.RandomState(0),
) -> datasets.Dataset:
    """Generate question by randomly"""
    lookups = _build_lookups(corpus, keys=["id", "kb_id", "answer_id"])
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

    return convert_rows_to_dataset(data)


def run():
    corpus = gen_corpus()
    rich.print(f"> corpus_hash: {corpus._fingerprint}")
    questions = _gen_questions(corpus)
    rich.print(f"> questions_hash: {questions._fingerprint}")

    lookup_pipe = pipes.LookupIndexPipe(corpus=corpus, n_sections=10)
    rich.print(f"> lookup_pipe_hash: {datasets.fingerprint.Hasher.hash(lookup_pipe)}")

    output = lookup_pipe(questions[:3])
    rich.print(output)
    pickle.dumps(lookup_pipe)


if __name__ == "__main__":
    run()
