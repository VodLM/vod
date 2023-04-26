from __future__ import annotations

from typing import Any, Literal

import datasets
import inflect

from raffle_ds_research.core.dataset_builders import retrieval_builder
from raffle_ds_research.tools.pipes.utils.misc import pack_examples

inflect_engine = inflect.engine()


class TestBuilderConfig(retrieval_builder.RetrievalBuilderConfig):
    """Configuration for the test builder."""

    name: Literal["test"] = "test"
    subset_name: Any = None


class TestBuilder(retrieval_builder.RetrievalBuilder):
    """Generates a dataset with random data for testing purposes."""

    n_points: int = 1000

    def _build_dset(self) -> datasets.DatasetDict:
        rows = self._gen_data("questions")
        return datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(pack_examples(rows)),
                "validation": datasets.Dataset.from_dict(pack_examples(rows)),
            }
        )

    def get_corpus(self) -> datasets.Dataset:
        """Returns the document sections for this dataset."""
        rows = self._gen_data("sections")
        return datasets.Dataset.from_dict(pack_examples(rows))

    def _gen_data(self, data_type: Literal["questions", "sections"]) -> list[dict[str, Any]]:
        return [_gen_row(i, data_type) for i in range(self.n_points)]


def _gen_row(i: int, data_type: Literal["questions", "sections"]) -> dict[str, Any]:
    if data_type == "questions":
        return retrieval_builder.QuestionModel(
            id=i,
            question=f"{inflect_engine.number_to_words(i)}",
            section_id=i if i % 2 else None,
            answer_id=i // 5,
            kb_id=i // 30,
        ).dict()

    if data_type == "sections":
        return retrieval_builder.SectionModel(
            section=f"{inflect_engine.number_to_words(i)}",
            id=i,
            answer_id=i // 5,
            kb_id=i // 30,
        ).dict()

    raise ValueError(f"Unknown data type: {data_type}")
