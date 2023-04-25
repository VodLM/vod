from typing import Optional, Protocol, Union

import datasets
from datasets import Dataset as HfDataset
from datasets import DatasetDict as HfDatasetDict

from .frank import FrankSplitName, HfFrankPart, load_frank

FRANK_SPLITS = [x.value for x in FrankSplitName]
FRANK_TYPES = ["qa_splits", "sections"]
LANGUAGE_CODES = ["en", "es", "de", "fr", "it", "nl", "pt", "ru", "tr", "zh", "da", "nl"]


class DatasetLoaderFn(Protocol):
    """A Factory method to load a Raffle dataset."""

    def __call__(
        self,
        name: str,
        cache_dir: Optional[str],
        keep_in_memory: Optional[bool],
    ) -> Union[HfDataset, HfDatasetDict]:
        """Load the dataset."""
        ...


def load_raffle_dataset(
    path: str = "frank",
    name: str = "en.A.qa_splits",
    split: Optional[datasets.Split] = None,
    cache_dir: Optional[str] = None,
    keep_in_memory: Optional[bool] = None,
) -> Union[HfDataset, HfDatasetDict]:
    """Load a dataset from the local cache or download it if it's not available."""
    known_loaders: dict[str, DatasetLoaderFn] = {
        "frank": _load_dataset_frank,
    }
    if path not in known_loaders:
        raise ValueError(f"Unknown path: {path}. Must be one of {known_loaders.keys()}.")

    # load the dataset
    loader = known_loaders[path]
    dset = loader(name, cache_dir=cache_dir, keep_in_memory=keep_in_memory)

    if split is not None:
        if not isinstance(dset, HfDatasetDict):
            raise ValueError(f"Cannot select a split from a dataset of type {type(dset)}")
        dset = dset[split]

    assert isinstance(dset, (HfDatasetDict, HfDataset))
    return dset


def _load_dataset_frank(
    name: str,
    cache_dir: Optional[str],
    keep_in_memory: Optional[bool],
) -> Union[HfDataset, HfDatasetDict]:
    language, frank_split, type_ = name.split(".")
    if frank_split not in FRANK_SPLITS:
        raise ValueError(
            f"Invalid split name: {frank_split}. Must be one of {FRANK_SPLITS}. "
            f"Name should be of the form `language.split.type`."
        )
    if type_ not in FRANK_TYPES:
        raise ValueError(
            f"Invalid type: {type_}. Must be one of {FRANK_TYPES} " f"Name should be of the form `language.split.type`."
        )
    frank: HfFrankPart = load_frank(language, split=frank_split, cache_dir=cache_dir, keep_in_memory=keep_in_memory)
    dset = {
        "qa_splits": frank.qa_splits,
        "sections": frank.sections,
    }

    return dset[type_]
