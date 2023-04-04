from typing import Optional

import datasets
import pydantic

from raffle_ds_research.tools.raffle_datasets.base import RetrievalDataset
from raffle_ds_research.tools.raffle_datasets.frank import load_frank
from raffle_ds_research.tools.raffle_datasets.msmarco import load_msmarco
from raffle_ds_research.tools.raffle_datasets.squad import load_squad


class DatasetLoader(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    name: str
    language: str = "en"
    subset_name: Optional[str] = None
    cache_dir: Optional[str] = None
    invalidate_cache: bool = False
    keep_in_memory: Optional[bool] = None
    only_positive_sections: bool = False

    def __call__(self) -> RetrievalDataset:
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


class ConcatenatedDatasetLoader:
    def __init__(self, loaders: list[DatasetLoader]):
        self.loaders = loaders

    def __call__(self) -> RetrievalDataset:
        dsets = [loader() for loader in self.loaders]

        # concatenate all sections
        all_sections = datasets.concatenate_datasets([dset.sections for dset in dsets])

        # concatenate all qa splits
        all_qa_splits = {}
        for split in {split for dset in dsets for split in dset.qa_splits.keys()}:
            split_dsets = [dset.qa_splits[split] for dset in dsets if split in dset.qa_splits]
            all_qa_splits[split] = datasets.concatenate_datasets(split_dsets)

        return RetrievalDataset(sections=all_sections, qa_splits=datasets.DatasetDict(all_qa_splits))
