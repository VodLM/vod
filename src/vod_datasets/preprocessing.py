import functools
import hashlib
import pickle
from typing import Any, Optional

import datasets
import numpy as np
from vod_tools import pipes

from src import vod_configs


def preprocess_queries(
    dset: datasets.Dataset,
    config: vod_configs.BaseDatasetConfig,
    info: str,
) -> datasets.Dataset:
    """Preprocess `queries` dataset."""
    dset = dset.map(
        pipes.Partial(
            pipes.template_pipe,
            template=config.options.templates.queries,
            input_keys=["query", "language"],
            output_key="text",
        ),
        **_prep_map_kwargs(config.options.prep_map_kwargs, desc=f"{info}: Preprocessing questions"),
    )

    return _shared_preprocessing(dset, config, info)


def preprocess_sections(
    dset: datasets.Dataset,
    config: vod_configs.BaseDatasetConfig,
    info: str,
) -> datasets.Dataset:
    """Preprocess `queries` dataset."""
    dset = dset.map(
        pipes.Partial(
            pipes.template_pipe,
            template=config.options.templates.sections,
            input_keys=["title", "content", "language"],
            output_key="text",
        ),
        **_prep_map_kwargs(config.options.prep_map_kwargs, desc=f"{info}: Preprocessing sections"),
    )
    return _shared_preprocessing(dset, config, info)


def _shared_preprocessing(dset: datasets.Dataset, config: vod_configs.BaseDatasetConfig, info: str) -> datasets.Dataset:
    dset = _take_subset(dset, config.options.subset_size)
    dset = _add_extras_attributes(dset, config, info)
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


def combine_datasets(
    inputs: datasets.Dataset | list[datasets.Dataset | datasets.DatasetDict],
) -> datasets.Dataset:
    """Combine a list of datasets into a single dataset."""
    if isinstance(inputs, datasets.Dataset):
        return inputs

    if isinstance(inputs, list):
        inputs = [combine_datasets(d) for d in inputs]  # type: ignore
        return datasets.concatenate_datasets(inputs)  # type: ignore

    if isinstance(inputs, (datasets.DatasetDict, dict)):
        return combine_datasets(list(inputs.values()))

    raise TypeError(f"Unexpected type `{type(inputs)}`")
