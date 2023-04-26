from __future__ import annotations

import functools
from typing import Any, Optional, TypeVar

import datasets
import pydantic

from raffle_ds_research.tools.raffle_datasets.base import RetrievalDataset
from raffle_ds_research.tools.raffle_datasets.frank import load_frank
from raffle_ds_research.tools.raffle_datasets.msmarco import load_msmarco
from raffle_ds_research.tools.raffle_datasets.squad import load_squad


class DatasetLoader(pydantic.BaseModel):
    """A Factory for loading datasets."""

    class Config:
        """Pydantic configuration."""

        extra = pydantic.Extra.forbid

    name: str
    language: str = "en"
    subset_name: Optional[str] = None
    cache_dir: Optional[str] = None
    invalidate_cache: bool = False
    keep_in_memory: Optional[bool] = None
    only_positive_sections: bool = False

    def __call__(self) -> RetrievalDataset:
        """Load the dataset."""
        loader = {
            "frank": load_frank,
            "msmarco": load_msmarco,
            "squad": load_squad,
        }[self.name]

        return loader(
            subset_name=self.subset_name,
            language=self.language,
            cache_dir=self.cache_dir,
            invalidate_cache=self.invalidate_cache,
            keep_in_memory=self.keep_in_memory,
            only_positive_sections=self.only_positive_sections,
        )


def _get_unique_values(dset: datasets.Dataset, column: str, chunk: int = 1_000) -> set[Any]:
    values = set()
    for i in range(0, len(dset), chunk):
        rows = dset[i : i + chunk]
        values.update(set(rows[column]))

    return values


DorDD = TypeVar("DorDD", datasets.Dataset, datasets.DatasetDict)


def _add_dset_idx(_: dict, dset_idx: int, key: str = "dset_idx") -> dict:
    return {key: dset_idx}


def _get_fingerprints(dset: datasets.Dataset | datasets.DatasetDict) -> str | dict[str, str]:
    if isinstance(dset, datasets.DatasetDict):
        return {k: v._fingerprint for k, v in dset.items()}

    return dset._fingerprint


class ConcatenatedDatasetLoader:
    """A class to concatenate multiple datasets together."""

    def __init__(self, loaders: list[DatasetLoader]):
        self.loaders = loaders

    def __call__(self) -> RetrievalDataset:
        """Load the datasets and concatenate them together."""
        dsets = [loader() for loader in self.loaders]

        # check there is no overlap in the kb_ids between the datasets
        kb_ids = [_get_unique_values(dset.sections, "kb_id") for dset in dsets]
        if len(set.union(*kb_ids)) != sum(len(ids) for ids in kb_ids):
            raise ValueError("There is overlap in the `kb_ids`")

        # add the dataset idx to the questions
        # todo: don't add this manually, instead add a `dataset_name` column in the loader.
        for idx, dset in enumerate(dsets):
            fn = functools.partial(_add_dset_idx, dset_idx=idx, key="dset_idx")
            dset.sections = dset.sections.map(fn, desc="Adding `dset_idx` to sections", num_proc=4)
            dset.qa_splits = dset.qa_splits.map(fn, desc="Adding `dset_idx` to questions", num_proc=4)

        # concatenate all sections
        all_sections = datasets.concatenate_datasets([dset.sections for dset in dsets])

        # concatenate all qa splits
        all_qa_splits = {}
        for split in {split for dset in dsets for split in dset.qa_splits}:
            split_dsets = [dset.qa_splits[split] for dset in dsets if split in dset.qa_splits]
            all_qa_splits[split] = datasets.concatenate_datasets(split_dsets)

        return RetrievalDataset(sections=all_sections, qa_splits=datasets.DatasetDict(all_qa_splits))
