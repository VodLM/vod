import dataclasses
import functools
import typing

import datasets
import xxhash
from vod_datasets.rosetta import models

T = typing.TypeVar("T")


def _compute_section_hash(section: models.SectionModel) -> str:
    """Get the hash of the section."""
    hasher = xxhash.xxh128()
    hasher.update(section.content)
    if section.title is not None:
        hasher.update(section.title)
    return hasher.hexdigest()


@dataclasses.dataclass
class QueriesWithSections(typing.Generic[T]):
    """A retrieval dataset of queries paired with sections."""

    queries: T
    sections: T

    def __getitem__(self, idx: str) -> T:
        """Get an item from the dataset."""
        try:
            return getattr(self, idx)
        except AttributeError as exc:
            raise KeyError(f"Invalid key `{idx}`. Valid keys are `{list(self.__dict__.keys())}`") from exc


def _is_idx_in(row: dict, idx: int, *, allowed_ids: list[int]) -> bool:  # noqa: ARG001
    """Check if an index is in a row."""
    return idx in allowed_ids


def _extract_sections(batch: dict[str, list[typing.Any]], idx: list[int]) -> dict[str, list[typing.Any]]:
    """Extract sections from a batch."""
    keys = list(batch.keys())
    flattened_sections = []
    for t, _ in enumerate(idx):
        row = {k: batch[k][t] for k in keys}
        row_ = models.QueryWithContextsModel(**row)
        for j, content in enumerate(row_.contexts):
            title = row_.titles[j] if row_.titles is not None else None
            flattened_sections.append(
                models.SectionModel(
                    content=content,
                    title=title,
                    language=row_.language,
                )
            )

    return {
        "content": [s.content for s in flattened_sections],
        "title": [s.title for s in flattened_sections],
        "language": [s.language for s in flattened_sections],
        "subset_id": [_compute_section_hash(s) for s in flattened_sections],
    }


def _extract_queries_and_assign_subset_ids(row: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Assign a subset ID to a row."""
    row_ = models.QueryWithContextsModel(**row)
    sections = []
    for i, section in enumerate(row_.contexts):
        title = row_.titles[i] if row_.titles is not None else None
        sections.append(
            models.SectionModel(
                content=section,
                title=title,
                language=row_.language,
            )
        )
    return {"subset_ids": [_compute_section_hash(s) for s in sections]}


def isolate_qa_and_sections(
    data: datasets.Dataset | datasets.DatasetDict,
    num_proc: int = 4,
) -> QueriesWithSections[datasets.Dataset]:
    """Preprocess the data into a retrieval dataset.

    Each unique `context` will be converted into a section. The `subset_id` will be used to link queries to sections.
    """
    sections = data.map(
        _extract_sections,
        batched=True,
        with_indices=True,
        num_proc=num_proc,
        desc="Extracting sections",
        remove_columns=list(data.column_names),
    )

    # Filter out duplicate sections
    section_unique_ids: dict[str, int] = {}
    for idx, row in enumerate(sections):
        row_ = models.SectionModel(**row)  # type: ignore
        if row_.subset_id is None:
            raise ValueError(f"Row `{row}` does not have a subset ID.")
        if row_.subset_id not in section_unique_ids:
            section_unique_ids[row_.subset_id] = idx

    sections = sections.filter(
        functools.partial(_is_idx_in, allowed_ids=list(section_unique_ids.values())),
        num_proc=num_proc,
        with_indices=True,
        batched=False,
        desc="Filtering sections",
    )

    # Extract the queries and assign subset IDs
    queries = data.map(
        _extract_queries_and_assign_subset_ids,
        batched=False,
        with_indices=False,
        num_proc=num_proc,
        desc="Assigning subset IDs",
        remove_columns=[k for k in ["contexts", "titles"] if k in data.column_names],
    )

    # Clean up the queries and return
    return QueriesWithSections(
        queries=queries,
        sections=sections,
    )  # type: ignore
