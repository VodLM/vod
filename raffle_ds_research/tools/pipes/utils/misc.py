from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Sized, TypeVar

import numpy as np
from datasets import Dataset as HfDataset

T = TypeVar("T")


def unpack_batch(batch: dict[str, list], keys: list[str]) -> Iterable[dict]:
    subset = {key: batch[key] for key in keys}
    master_key, *other_keys = subset
    for i in range(len(batch[master_key])):
        example = {key: batch[key][i] for key in keys}
        yield example


def repack_examples(examples: Iterable[dict]) -> dict[str, list]:
    output = defaultdict(list)
    for example in examples:
        for key in example.keys():
            output[key].append(example[key])
    return output


def pad_list(
    x: list[T],
    length: int,
    fill_value: Optional[T] = None,
    fill_values: Optional[Sized[T]] = None,
) -> list[T]:
    if fill_values is not None and fill_value is not None:
        raise ValueError("`fill_value` and `fill_values` cannot be both set")
    n_missing = length - len(x)
    if n_missing <= 0:
        return x[:length]
    elif fill_values is None:
        return x + [fill_value] * n_missing
    else:
        if isinstance(fill_values, set):
            fill_values = list(fill_values)
        samples = np.random.choice(fill_values, n_missing, replace=n_missing > len(fill_values))
        samples = samples.tolist()
        return x + samples


def keep_only_columns(dataset: HfDataset, columns: Iterable[str], strict: bool = True):
    columns = set(columns)
    if strict and not columns.issubset(dataset.column_names):
        raise ValueError(
            f"Columns {columns - set(dataset.column_names)} not in dataset " f"and are required with strict=True"
        )
    cols_to_remove = set(dataset.column_names) - columns
    cols_to_remove = list(sorted(cols_to_remove))
    dataset = dataset.remove_columns(cols_to_remove)
    return dataset
