import collections
import math
import time
import typing as typ

import numpy as np
import torch

T = typ.TypeVar("T")

VECTOR_KEY: str = "__vector__"


class BlockTimer:
    """A context manager for timing code blocks."""

    def __init__(self, name: str, output: dict[str, typ.Any]) -> None:
        self.name = name
        self.output = output

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, *args: typ.Any) -> None:
        self.output[self.name] = time.perf_counter() - self.start


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
