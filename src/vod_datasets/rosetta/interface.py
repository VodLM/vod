import dataclasses
import functools
import hashlib
import typing

import datasets
import rich
from vod_datasets.models import ModelType, QueryWithContextModel
from vod_datasets.rosetta.input_models import WithContextMixin

from .adapters import Adapter, find_adapter, get_first_row

T = typing.TypeVar("T")


@dataclasses.dataclass
class RetrievalDataset(typing.Generic[T]):
    """A retrieval dataset of queries paired with sections."""

    queries: T
    sections: T

    def __getitem__(self, idx: str) -> T:
        """Get an item from the dataset."""
        return getattr(self, idx)


def translate(data: datasets.Dataset, output: ModelType, num_proc: int = 4) -> datasets.Dataset:
    """Translate a Huggingface daatset."""
    row = get_first_row(data)
    adapter: None | typing.Type[Adapter] = find_adapter(row, output="query_with_context")
    rich.print({"adapter": adapter, "row": row})

    # Process `QueryWithContext` datasets
    if adapter is not None:
        data = adapter.translate(data)
        data_split = _make_retrieval_dataset(data, num_proc=num_proc)
        return data_split[output]

    # Process `Query` or `Section` datasets
    adapter = find_adapter(row, output=output)
    if adapter is None:
        raise ValueError(f"Could not find an adapter for `{output}` = {row}")
    return adapter.translate(data)


def _is_idx_in(row: dict, idx: int, *, allowed_ids: list[int]) -> bool:  # noqa: ARG001
    """Check if an index is in a row."""
    return idx in allowed_ids


def _hash_section(row: WithContextMixin | QueryWithContextModel) -> str:
    """Hash a section."""
    txt = f"Title: {row.title} | Context: {row.section}"
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def _assign_section_ids(row: dict, idx: int, *, section_lookup: dict[str, int]) -> dict:  # noqa: ARG001
    """Assign section IDs to a row."""
    rich.print(row)
    rich.print(row.keys())
    rich.print(WithContextMixin.model_fields)
    row_hash = _hash_section(WithContextMixin(**row))
    row["section_ids"] = [section_lookup[row_hash]]
    return row


def _make_retrieval_dataset(
    data: datasets.Dataset,
    num_proc: int = 4,
) -> RetrievalDataset[datasets.Dataset]:
    """Preprocess the data into a retrieval dataset."""
    section_lookup: dict[str, int] = {}
    for idx, row in enumerate(data):
        row_ = QueryWithContextModel(**row)  # Validate the row # type: ignore
        row_hash = _hash_section(row_)
        if row_hash not in section_lookup:
            section_lookup[row_hash] = idx

    sections = data.filter(
        functools.partial(_is_idx_in, allowed_ids=list(section_lookup.values())),
        num_proc=num_proc,
        with_indices=True,
        batched=False,
        desc="Filtering sections",
    )
    sections = sections.remove_columns(["query", "answer"])

    # Assign section IDs to the queries
    query = data.map(
        functools.partial(_assign_section_ids, section_lookup=section_lookup),
        num_proc=num_proc,
        with_indices=True,
        batched=False,
        desc="Assigning section IDs",
    )
    query = query.remove_columns(["section", "title"])
    return RetrievalDataset(
        queries=query,
        sections=sections,
    )
