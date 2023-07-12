from __future__ import annotations

import json
import pathlib
import warnings
from typing import Any, Optional

import datasets
from raffle_ds_research.tools.pipes.utils.misc import pack_examples
from raffle_ds_research.tools.raffle_datasets import RetrievalDataset

SQUAD_KB_ID = 200_000


def _add_row_idx(_: dict, idx: int) -> dict[str, int]:
    return {"id": idx}


class SquadRetrievalDataset(RetrievalDataset):
    """The Squad dataset for retrieval."""

    ...


class AppendQaExtras:
    """Append the `kb_id`, `language` and `section_ids` to the QA row."""

    def __init__(self, lookup: dict[str, int], language: str):
        self.lookup = lookup
        self.language = language

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        """Append the extras to the QA row."""
        row["kb_id"] = SQUAD_KB_ID
        row["language"] = self.language
        row_id = row["context"]
        row["section_ids"] = [self.lookup[row_id]]
        return row


@datasets.fingerprint.hashregister(AppendQaExtras)
def _register_append_qa_extras(hasher: datasets.fingerprint.Hasher, value: AppendQaExtras) -> str:
    data = json.dumps({"lookup": value.lookup, "language": value.language})
    return hasher.hash(data)


def load_squad(
    language: str = "en",
    subset_name: Optional[str] = None,
    invalidate_cache: Optional[bool] = None,
    cache_dir: Optional[str | pathlib.Path] = None,
    keep_in_memory: Optional[bool] = None,
    only_positive_sections: Optional[bool] = None,
    kb_id: Optional[int] = None,  # noqa: ARG
    prep_num_proc: int = 4,
) -> SquadRetrievalDataset:
    """Load the Squad dataset."""
    if language != "en":
        raise ValueError(f"Language `{language}` not supported")

    if invalidate_cache is not None:
        warnings.warn(
            f"`invalidate_cache={invalidate_cache}` not supported for SQuAD and will be ignored.", stacklevel=2
        )

    if subset_name is not None:
        warnings.warn(f"`subset_name={subset_name}` not supported for SQuAD and will be ignored.", stacklevel=2)

    if only_positive_sections is not None:
        warnings.warn(
            f"`only_positive_sections={only_positive_sections}` not supported for SQuAD and will be ignored.",
            stacklevel=2,
        )

    qa_splits: datasets.DatasetDict = datasets.load_dataset(
        "squad",
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
    )  # type: ignore

    # fetch the sections
    lookup = {}
    sections = []
    section_id = 0
    for dset in qa_splits.values():
        for _i, row in enumerate(dset):
            row_id = row["context"]
            if row_id not in lookup:
                lookup[row_id] = section_id
                section = {
                    "title": row["title"],
                    "content": row["context"],
                    "language": language,
                    "id": section_id,
                    "kb_id": SQUAD_KB_ID,
                }
                sections.append(section)
                section_id += 1

    sections = datasets.Dataset.from_dict(pack_examples(sections))

    qa_splits = qa_splits.map(
        AppendQaExtras(lookup=lookup, language=language),
        num_proc=prep_num_proc,
        remove_columns=["title", "context"],
        desc="Looking up section ids",
    )
    qa_splits = qa_splits.rename_columns({"question": "text", "id": "uid"})
    qa_splits = qa_splits.map(_add_row_idx, num_proc=prep_num_proc, desc="Adding row indices", with_indices=True)

    return SquadRetrievalDataset(qa_splits=qa_splits, sections=sections)
