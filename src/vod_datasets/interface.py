from __future__ import annotations

import datasets
import loguru
from vod_datasets import rosetta

from src import vod_configs

from .preprocessing import (
    combine_datasets,
    preprocess_queries,
    preprocess_sections,
)


def _load_dataset(config: vod_configs.BaseDatasetConfig) -> datasets.Dataset:
    """Load the dataset, process it according to the prompt template and return a HF dataset."""
    loguru.logger.info("Loading dataset `{descriptor}`", descriptor=config.descriptor)
    loaded_subsets = [
        datasets.load_dataset(config.name_or_path, subset, split=config.split) for subset in config.subsets
    ]
    return combine_datasets(loaded_subsets)  # type: ignore


def load_queries(config: vod_configs.BaseDatasetConfig) -> datasets.Dataset:
    """Load a queries dataset."""
    dset = _load_dataset(config)
    dset = rosetta.transform(dset, output="queries")
    dset = preprocess_queries(dset, config=config, info=f"{config.descriptor}(queries)")
    return dset


def load_sections(config: vod_configs.BaseDatasetConfig) -> datasets.Dataset:
    """Load a sections dataset."""
    dset = _load_dataset(config)
    dset = rosetta.transform(dset, output="sections")
    dset = preprocess_sections(dset, config=config, info=f"{config.descriptor}(sections)")
    return dset


def load_dataset(config: vod_configs.BaseDatasetConfig) -> datasets.Dataset:
    """Load a dataset."""
    if isinstance(config, vod_configs.QueriesDatasetConfig):
        return load_queries(config)
    if isinstance(config, vod_configs.SectionsDatasetConfig):
        return load_sections(config)
    raise TypeError(f"Unexpected config type `{type(config)}`")
