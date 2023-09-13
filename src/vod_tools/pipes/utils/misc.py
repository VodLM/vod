from collections import defaultdict
from typing import Any, Iterable, Optional, TypeVar

import datasets
import numpy as np

T = TypeVar("T")
_FILL_VALUE_UNSET = object()


def iter_examples(
    batch: dict[str, list],
    keys: None | Iterable[str] = None,
    allow_missing: bool = False,
) -> Iterable[dict]:
    """Iterate over the examples contained in a batch."""
    input_keys = set(batch.keys())
    requested_keys = set(keys or input_keys)
    if not allow_missing and requested_keys > input_keys:
        raise ValueError(f"Expected keys {set(requested_keys)}, got {set(batch.keys())}")
    handled_keys = list(requested_keys.intersection(input_keys))
    for i in range(len(batch[handled_keys[0]])):
        example = {key: batch[key][i] for key in handled_keys}
        yield example


def pack_examples(examples: Iterable[dict[T, Any]], keys: None | list[T] = None) -> dict[T, list[Any]]:
    """Pack a list of examples into a batch."""
    output = defaultdict(list)
    for example in examples:
        keys_ = keys or set(example.keys())
        if not set(keys_).issubset(example.keys()):
            raise ValueError(f"Expected keys {set(keys_)}, got {set(example.keys())}")
        for key in keys_:
            output[key].append(example[key])
    return dict(output)


def pad_list(
    x: list[T],
    length: int,
    fill_value: T = _FILL_VALUE_UNSET,
    fill_values: Optional[list[T]] = None,
) -> list[T]:
    """Pad a list to a given length."""
    if fill_values is not None and fill_value is not None:
        raise ValueError("`fill_value` and `fill_values` cannot be both set")
    n_missing = length - len(x)
    if n_missing <= 0:
        return x[:length]

    if fill_values is None:
        if fill_value is _FILL_VALUE_UNSET:
            raise ValueError("Must set either `fill_value` or `fill_values`")
        return x + [fill_value] * n_missing

    if isinstance(fill_values, set):
        fill_values = list(fill_values)
    samples = np.random.choice(fill_values, n_missing, replace=n_missing > len(fill_values))
    samples = samples.tolist()
    return x + samples


def keep_only_columns(dataset: datasets.Dataset, columns: Iterable[str], strict: bool = True) -> datasets.Dataset:
    """Keep only the specified columns in a `datasets.Dataset`."""
    columns = set(columns)
    if strict and not columns.issubset(dataset.column_names):
        raise ValueError(
            f"Columns {columns - set(dataset.column_names)} not in dataset and are required with argument `strict=True`"
        )
    cols_to_remove = set(dataset.column_names) - columns
    cols_to_remove = sorted(cols_to_remove)
    return dataset.remove_columns(cols_to_remove)
