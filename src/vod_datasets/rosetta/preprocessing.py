import dataclasses
import re
import typing

import datasets
import xxhash
from datasets import fingerprint
from vod_datasets.rosetta import models

T = typing.TypeVar("T")
ALPHANUM_PATTERN = re.compile(r"[^\w]|[\s]")


def _compute_section_hash(content: str, title: None | str) -> str:
    """Get the hash of the section."""
    hasher = xxhash.xxh128()
    hasher.update(content)
    if title is not None:
        hasher.update(title)
    return f"{hasher.hexdigest()}-{len(content)}"  # add the length to be extra careful with collisions


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


class _ExtractSections:
    query_with_context_model: typing.Type[models.QueryWithContextsModel]
    section_model: typing.Type[models.SectionModel]

    def __init__(
        self,
        query_with_context_model: typing.Type[models.QueryWithContextsModel],
        section_model: typing.Type[models.SectionModel],
    ) -> None:
        """Initialize the class."""
        self.query_with_context_model = query_with_context_model
        self.section_model = section_model

    """A utility function to extract sections."""

    def __call__(self, batch: dict[str, list[typing.Any]], idx: list[int]) -> dict[str, list[typing.Any]]:
        """Extract sections from a batch."""
        keys = list(batch.keys())
        flattened_sections = []
        for t, _ in enumerate(idx):
            row = {k: batch[k][t] for k in keys}
            row_ = self.query_with_context_model(**row)
            for j, content in enumerate(row_.contexts):
                title = row_.titles[j] if row_.titles is not None else None
                section_hash = _compute_section_hash(content, title)
                flattened_sections.append(
                    self.section_model(
                        content=content,
                        title=title,
                        id=section_hash,
                        subset_id=section_hash,
                    )
                )

        return {
            "content": [s.content for s in flattened_sections],
            "title": [s.title for s in flattened_sections],
            "id": [s.id for s in flattened_sections],
            "subset_id": [s.subset_id for s in flattened_sections],
        }


@fingerprint.hashregister(_ExtractSections)
def _hash_extract_sections(hasher: datasets.fingerprint.Hasher, obj: _ExtractSections) -> str:
    """Register the `_IsIdxIn` class to work with `datasets.map()`."""
    return hasher.hash(
        {
            "cls": obj.__class__,
            "query_with_context_model": obj.query_with_context_model,
            "section_model": obj.section_model,
        }
    )


def _extract_queries_and_assign_subset_ids(row: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Assign a subset ID to a row."""
    row_ = models.QueryWithContextsModel(**row)
    sections = []
    for i, section in enumerate(row_.contexts):
        title = row_.titles[i] if row_.titles is not None else None
        section_hash = _compute_section_hash(section, title)
        sections.append(
            models.SectionModel(
                content=section,
                title=title,
                id=section_hash,
                subset_id=section_hash,
            )
        )
    return {"subset_ids": [s.subset_id for s in sections]}


class _UniqueValuesMap:
    """A utility function to keep one row per unique value.

    The `_unique_ids_map` is built lazily so the whole operation can be hashed and cached by `datasets.map()`.
    """

    _unique_ids_map: None | list[int]
    sections: datasets.Dataset

    def __init__(self, sections: datasets.Dataset, *, key: str) -> None:
        self.key = key
        self.sections = sections
        self._unique_ids_map = None

    @staticmethod
    def _build_allowed_ids(sections: datasets.Dataset, key: str) -> list[int]:
        """Build the allowed IDs."""
        unique_values_ids: dict[str, int] = {}
        for idx, row in enumerate(sections):
            value = row.get(key, None)
            if value is None:
                raise ValueError(f"Row `{row}` does not have a subset ID.")
            if value not in unique_values_ids:
                unique_values_ids[value] = idx
        return list(unique_values_ids.values())

    def __call__(self, row: dict, idx: int) -> bool:  # noqa: ARG
        """Check if an index is in a row."""
        if self._unique_ids_map is None:
            self._unique_ids_map = self._build_allowed_ids(self.sections, self.key)
        return idx in self._unique_ids_map


@fingerprint.hashregister(_UniqueValuesMap)
def _hash_unique_values_map(hasher: datasets.fingerprint.Hasher, obj: _UniqueValuesMap) -> str:
    """Register the `_UniqueValuesMap` class to work with `datasets.map()`."""
    return hasher.hash(
        {
            "cls": obj.__class__,
            "key": obj.key,
            "sections": obj.sections._fingerprint,
        }
    )


def isolate_qa_and_sections(
    data: datasets.Dataset,
    num_proc: int = 4,
) -> QueriesWithSections[datasets.Dataset]:
    """Preprocess the data into a retrieval dataset.

    Each unique `context` will be converted into a section.
    The `subset_id` will be used to link queries to sections.
    NB: make sure to sort the `remove_columns` to ensure the fingerprint is consistent.
    """
    sections = data.map(
        _ExtractSections(
            query_with_context_model=models.QueryWithContextsModel,
            section_model=models.SectionModel,
        ),
        batched=True,
        with_indices=True,
        num_proc=num_proc,
        desc="Extracting sections",
        remove_columns=sorted(set(data.column_names) - {"id"}),
    )

    sections = sections.filter(
        _UniqueValuesMap(sections, key="subset_id"),
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
        remove_columns=sorted([k for k in ["contexts", "titles"] if k in data.column_names]),
    )

    # Clean up the queries and return
    return QueriesWithSections(
        queries=queries,
        sections=sections,
    )  # type: ignore
