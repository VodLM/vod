import collections
import math
import time
import typing as typ
import warnings

import numpy as np
import torch
import vod_search

T = typ.TypeVar("T")


class BlockTimer:
    """A context manager for timing code blocks."""

    def __init__(self, name: str, output: dict[str, typ.Any]) -> None:
        self.name = name
        self.output = output

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, *args: typ.Any) -> None:
        self.output[self.name] = time.perf_counter() - self.start


def fill_nans_with_min(values: np.ndarray, offset_min_value: None | float = -1, axis: int = -1) -> np.ndarray:
    """Replace NaNs with the minimum value along each dimension."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        min_scores = np.nanmin(values, axis=axis, keepdims=True)
        min_scores = np.where(np.isnan(min_scores), 0, min_scores)
        if offset_min_value is not None:
            min_scores += offset_min_value  # make sure the min is lower than the rest
        return np.where(np.isnan(values), min_scores, values)


def replace_negative_indices(sections: vod_search.RetrievalBatch, world_size: int) -> vod_search.RetrievalBatch:
    """Replace negative indices with random ones."""
    is_negative = sections.indices < 0
    n_negative = is_negative.sum()
    if n_negative:
        sections.indices.setflags(write=True)
        sections.indices[is_negative] = np.random.randint(0, world_size, size=n_negative)
    return sections


def reshape_flat_list(lst: list[T], shape: tuple[int, int]) -> list[list[T]]:
    """Reshape a list."""
    if len(shape) != 2:  # noqa: PLR2004
        raise ValueError(f"Expected a 2D shape. Found {shape}")
    if math.prod(shape) != len(lst):
        raise ValueError(f"Expected a list of length {math.prod(shape)}. Found {len(lst)}")
    return [lst[i : i + shape[1]] for i in range(0, len(lst), shape[1])]


def cast_as_tensor(
    x: list | np.ndarray | torch.Tensor,
    dtype: torch.dtype,
    replace: None | dict[T, T] = None,
) -> torch.Tensor:
    """Cast a value as a torch.Tensor."""
    if replace is not None:
        if isinstance(x, list):
            x = [replace.get(i, i) for i in x]
        else:
            raise TypeError(f"Cannot use `replace` with type {type(x)}")

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(dtype=dtype)
    elif isinstance(x, torch.Tensor):
        x = x.to(dtype=dtype)
    else:
        x = torch.tensor(x, dtype=dtype)

    return x


V = typ.TypeVar("V")


def pack_examples(
    examples: typ.Iterable[dict[T, V]],
    keys: None | list[T] = None,
) -> dict[T, list[V]]:
    """Pack a list of examples into a batch."""
    output = collections.defaultdict(list)
    for example in examples:
        keys_ = keys or set(example.keys())
        if not set(keys_).issubset(example.keys()):
            raise ValueError(f"Expected keys {set(keys_)}, got {set(example.keys())}")
        for key in keys_:
            output[key].append(example[key])

    return dict(output)
