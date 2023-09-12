from typing import Any

import datasets
import numpy as np

from src import vod_configs


def postprocess_queries(
    dset: datasets.Dataset,
    identifier: str,
    config: vod_configs.DatasetOptions,
) -> datasets.Dataset:
    """Post-process `queries` dataset. E.g., format `text` using `template`."""
    # Common post-processing (add extra attributes, etc.)
    output = _postprocessing(dset, identifier=identifier, config=config)
    return output


def postprocess_sections(
    dset: datasets.Dataset,
    identifier: str,
    config: vod_configs.DatasetOptions,
) -> datasets.Dataset:
    """Post-process `queries` dataset."""
    # TODO: sectioning
    output = _postprocessing(dset, identifier=identifier, config=config)
    return output


def _postprocessing(
    dset: datasets.Dataset,
    identifier: str,  # noqa: ARG
    config: vod_configs.DatasetOptions,
) -> datasets.Dataset:
    dset = _take_subset(dset, config.subset_size)
    return dset


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
