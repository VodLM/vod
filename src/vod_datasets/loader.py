from __future__ import annotations

import functools
import hashlib
import pickle
from typing import Any, Optional

import datasets
import loguru
import numpy as np
from vod_datasets import rosetta
from vod_tools import pipes

from src import vod_configs

def combine_datasets(dataset_list: list[datasets.Dataset | datasets.DatasetDict]) -> datasets.Dataset | datasets.DatasetDict:
    """Combine a list of datasets into a single dataset."""

    # If all items are of type Dataset
    if all(isinstance(ds, datasets.Dataset) for ds in dataset_list):
        return datasets.concatenate_datasets(dataset_list) # type: ignore

    # If all items are of type DatasetDict
    elif all(isinstance(ds, datasets.DatasetDict) for ds in dataset_list):
        combined_dict = datasets.DatasetDict()
        for ds in dataset_list:
            for key, value in ds.items(): # type: ignore
                if key in combined_dict:
                    combined_dict[key] = datasets.concatenate_datasets([combined_dict[key], value])
                else:
                    combined_dict[key] = value
        return combined_dict

    # If there's a mix of Dataset and DatasetDict
    else:
        raise NotImplementedError("Combining a mix of Dataset and DatasetDict is not yet supported.")

def _load_dataset(config: vod_configs.BaseDatasetConfig) -> datasets.Dataset | datasets.DatasetDict:
    """Load the dataset, process it according to the prompt template and return a HF dataset."""
    loguru.logger.info("Loading the dataset: `{descriptor}`", descriptor=config.descriptor)
    loaded_subsets = [datasets.load_dataset(config.name_or_path, subset, split=config.split) for subset in config.subsets]
    return combine_datasets(loaded_subsets) # type: ignore


def load_queries(
    config: vod_configs.BaseDatasetConfig,
) -> datasets.DatasetDict:
    """Load a queries dataset."""
    dset = _load_dataset(config)
    dset = rosetta.transform(dset, output="query")
    # dset = _preprocess_queries(dset, config=config, locator=f"{config.descriptor}(queries)")
    return dset


def load_sections(
    config: vod_configs.BaseDatasetConfig,
) -> datasets.DatasetDict:
    """Load a sections dataset."""
    dset = _load_dataset(config)
    dset = rosetta.transform(dset, output="section")
    # dset = _preprocess_sections(dset, config=config, locator=f"{config.descriptor}(sections)")
    return dset


def load_dataset(config: vod_configs.BaseDatasetConfig) -> datasets.Dataset:
    """Load a dataset."""
    if isinstance(config, vod_configs.QueriesDatasetConfig):
        return load_queries(config)
    if isinstance(config, vod_configs.SectionsDatasetConfig):
        return load_sections(config)
    raise TypeError(f"Unexpected config type `{type(config)}`")


def _preprocess_queries(
    dset: datasets.Dataset,
    config: vod_configs.BaseDatasetConfig,
    locator: str,
) -> datasets.Dataset:
    dset = dset.map(
        pipes.Partial(
            pipes.template_pipe,
            template=config.templates.queries,
            input_keys=["query", "language", "kb_id"],
            output_key="text",
        ),
        **_prep_map_kwargs(config.options.prep_map_kwargs, desc=f"{locator}: Preprocessing questions"),
    )

    return _shared_preprocessing(dset, config, locator)


def _preprocess_sections(
    dset: datasets.Dataset,
    config: vod_configs.BaseDatasetConfig,
    locator: str,
) -> datasets.Dataset:
    dset = dset.map(
        pipes.Partial(
            pipes.template_pipe,
            template=config.templates.sections,
            input_keys=["title", "section", "language", "kb_id"],
            output_key="text",
        ),
        **_prep_map_kwargs(config.options.prep_map_kwargs, desc=f"{locator}: Preprocessing sections"),
    )
    return _shared_preprocessing(dset, config, locator)


def _shared_preprocessing(
    dset: datasets.Dataset, config: vod_configs.BaseDatasetConfig, locator: str
) -> datasets.Dataset:
    dset = _compute_group_hashes(
        dset,
        keys=config.options.group_keys,
        output_key=config.options.group_hash_key,
        **_prep_map_kwargs(config.options.prep_map_kwargs, desc=f"{locator}: Computing group hashes", batched=False),
    )
    dset = _take_subset(dset, config.options.subset_size)
    dset = _add_extras_attributes(dset, config, locator)
    return dset


def _add_extras(row: dict[str, Any], *args: Any, extras: dict[str, Any], **kwds: Any) -> dict[str, Any]:  # noqa: ARG001
    return extras


def _add_extras_attributes(
    dataset: datasets.Dataset,
    config: vod_configs.BaseDatasetConfig,
    locator: str,
) -> datasets.Dataset:
    """Add a descriptor to the dataset."""
    xtras = {"dset_uid": config.descriptor}
    if isinstance(config, vod_configs.QueriesDatasetConfig):
        xtras["link"] = config.link  # type: ignore

    return dataset.map(
        functools.partial(_add_extras, extras=xtras),
        **_prep_map_kwargs({}, desc=f"{locator}: adding extras attributess", batched=False),
    )


def _prep_map_kwargs(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    always_on = {"batched": True, "with_indices": True}
    return {**base, **always_on, **overrides}


def _take_subset(dset: datasets.Dataset, subset_size: None | int) -> datasets.Dataset:
    """Take a subset of the dataset."""
    if subset_size is None:
        return dset

    rgn = np.random.RandomState(0)

    # sample the subsets
    ids = rgn.choice(list(range(len(dset))), size=subset_size, replace=False)
    return dset.select(ids)


class KeyHasher:
    """Hash key values into a single `int64`."""

    def __init__(self, keys: list[str], output_key: str = "group_hash") -> None:
        self.keys = keys
        self.output_key = output_key

    def __call__(self, row: dict[str, Any], idx: Optional[int] = None, **kwds: Any) -> dict[str, Any]:  # noqa: ARG002
        """Hash the keys."""
        subrow = [(k, row[k]) for k in sorted(self.keys)]
        h = hashlib.sha256(pickle.dumps(subrow, 1))
        obj_hash = h.hexdigest()
        np_int_hash = np.int64(int(obj_hash, 16) % np.iinfo(np.int64).max)
        return {self.output_key: np_int_hash}


def _compute_group_hashes(
    dataset: datasets.Dataset,
    keys: list[str],
    output_key: str,
    **kws: Any,
) -> datasets.Dataset:
    """Compute group hashes based on some list of `keys`."""
    hasher = KeyHasher(keys=keys, output_key=output_key)
    return dataset.map(hasher, **kws)
