from __future__ import annotations

import re
import time
import warnings
from typing import Any, Optional, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def fill_nans_with_min(values: np.ndarray, offset_min_value: None | float = -1, axis: int = -1) -> np.ndarray:
    """Replace NaNs with the minimum value along each dimension."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        min_scores = np.nanmin(values, axis=axis, keepdims=True)
        min_scores = np.where(np.isnan(min_scores), 0, min_scores)
        if offset_min_value is not None:
            min_scores += offset_min_value  # make sure the min is lower than the rest
        return np.where(np.isnan(values), min_scores, values)


_clean_query_ptrn = re.compile(r"^(q:|query:|question:|question\s+text:|question\s+text\s+is:)\s*", re.IGNORECASE)
_strip_punctuation_ptrn = re.compile(r"[^\w\s]")
_subsequent_whitespaces_ptrn = re.compile(r"\s+")


def exact_match(q: str, d: str) -> bool:
    """Exact match between query and document."""
    q = _clean_query_ptrn.sub("", q)
    q = _strip_punctuation_ptrn.sub(" ", q).lower()
    d = _strip_punctuation_ptrn.sub(" ", d).lower()
    q = _subsequent_whitespaces_ptrn.sub(" ", q)
    d = _subsequent_whitespaces_ptrn.sub(" ", d)
    import rich

    rich.print({"q": q, "d": d})
    return q in d


class BlockTimer:
    """A context manager for timing code blocks."""

    def __init__(self, name: str, output: dict[str, Any]) -> None:
        self.name = name
        self.output = output

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, *args: Any) -> None:
        self.output[self.name] = time.perf_counter() - self.start


def cast_as_tensor(
    x: list | np.ndarray | torch.Tensor,
    dtype: torch.dtype,
    replace: Optional[dict[T, T]] = None,
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
