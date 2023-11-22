import collections
from typing import Any

import datasets
import numpy as np
import vod_configs
from vod_configs.misc.template import Template

from .sectioning import Sectionizer, init_sectionizer


def postprocess_queries(
    dset: datasets.Dataset,
    identifier: str,
    config: vod_configs.DatasetOptions,
) -> datasets.Dataset:
    """Post-process `queries` dataset. E.g., format `text` using `template`."""
    dset = _postprocessing(dset, identifier=identifier, config=config)
    return dset


def postprocess_sections(
    dset: datasets.Dataset,
    identifier: str,
    config: vod_configs.DatasetOptions,
) -> datasets.Dataset:
    """Post-process `queries` dataset."""
    if config.sectioning is not None:
        dset = _extract_sections(
            dset,
            config=config.sectioning,
            map_kwargs=_prep_map_kws(
                base=config.prep_map_kws,
                batched=True,
                desc=f"Extracting sections for `{identifier}`",
            ),
        )
    dset = _postprocessing(dset, identifier=identifier, config=config)
    return dset


def _section_extractor(
    batch: dict[str, list[Any]],
    idx: None | list[int] = None,  # noqa: ARG001
    *,
    sectionizer: Sectionizer,
    template: Template,
) -> dict[str, list[Any]]:
    contents = batch["content"]
    new_batch = collections.defaultdict(list)
    for i, content in enumerate(contents):
        # Render a section without content to infer how many tokens the prefix takes up
        rendered_section_zero = template.render({"content": "", "title": batch["title"][i]})
        for chunk in sectionizer(content, prefix=rendered_section_zero, add_prefix=False):
            new_batch["content"].append(chunk)
            for key in batch.keys() - {"content"}:
                new_batch[key].append(batch[key][i])

    return new_batch


def _extract_sections(
    data: datasets.Dataset,
    *,
    config: vod_configs.SectioningConfig,
    map_kwargs: None | dict[str, Any] = None,
) -> datasets.Dataset:
    sectionizer = init_sectionizer(config)
    template = Template(config.section_template)
    return data.map(
        _section_extractor,
        remove_columns=data.column_names,
        fn_kwargs={"sectionizer": sectionizer, "template": template},
        **(map_kwargs or {}),
    )


def _postprocessing(
    dset: datasets.Dataset,
    identifier: str,  # noqa: ARG001
    config: vod_configs.DatasetOptions,
) -> datasets.Dataset:
    dset = _take_subset(dset, config.subset_size)
    return dset


def _prep_map_kws(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
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
