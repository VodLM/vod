from __future__ import annotations

import functools
from numbers import Number
from typing import Any, Iterable, Optional

import numpy as np
import pydantic
import rich
import rich.console
import rich.table
import torch


_PPRINT_DISPLAY_PREC = 2


class Properties(pydantic.BaseModel):
    py_type: type = list
    shape: Optional[str] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    mean: Optional[str] = None
    min: Optional[str] = None
    max: Optional[str] = None


@functools.singledispatch
def infer_properties(x: Any) -> Properties:
    return Properties(py_type=type(x))


def _smart_str(x: Number) -> str:
    if isinstance(x, float):
        return f"{x:.{_PPRINT_DISPLAY_PREC}e}"
    elif isinstance(x, int):
        return f"{x}"
    elif isinstance(x, complex):
        return f"{x.real:.{_PPRINT_DISPLAY_PREC}e} + {x.imag:.{_PPRINT_DISPLAY_PREC}e}j"
    else:
        return str(x)


@infer_properties.register(torch.Tensor)
def _(x: torch.Tensor) -> Properties:
    xf = x.detach().float()
    return Properties(
        py_type=type(x),
        shape=str(list(x.shape)),
        dtype=str(x.dtype).replace("torch.", ""),
        device=str(x.device),
        mean=f"{_smart_str(xf.mean().item())}",
        min=f"{_smart_str(x.min().item())}",
        max=f"{_smart_str(x.max().item())}",
    )


@infer_properties.register(np.ndarray)
def _(x: np.ndarray) -> Properties:
    xf: np.ndarray = x.astype(np.float32)
    return Properties(
        py_type=type(x),
        shape=str(x.shape),
        dtype=str(x.dtype),
        mean=f"{_smart_str(xf.mean())}",
        min=f"{_smart_str(np.min(x))}",
        max=f"{_smart_str(np.max(x))}",
    )


@infer_properties.register(list)
@infer_properties.register(set)
@infer_properties.register(tuple)
def _(x: list | set | tuple) -> Properties:
    try:
        arr = np.array(x)
        shape = str(arr.shape)
    except Exception:
        shape = f"[{len(x)}, ?]"

    leaves_types = list({type(y) for y in _iter_leaves(x)})
    if all(issubclass(t, Number) for t in leaves_types):
        leaves_mean = np.mean(y for y in _iter_leaves(x))
        leaves_min = min(_iter_leaves(x))
        leaves_max = max(_iter_leaves(x))
    else:
        leaves_mean = "-"
        leaves_min = "-"
        leaves_max = "-"

    leaves_types = [t.__name__ for t in leaves_types]
    if len(leaves_types) == 1:
        leaves_types = leaves_types[0]

    return Properties(
        py_type=type(x),
        dtype=f"py[{leaves_types}]",
        shape=shape,
        min=leaves_min,
        max=leaves_max,
        device="-",
        mean=leaves_mean,
    )


def pprint_batch(
    batch: dict[str, Any],
    idx: Optional[list[int]] = None,
    console: Optional[rich.console.Console] = None,
    header: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    table = rich.table.Table(title=header, show_header=True, header_style="bold magenta")
    fields = list(Properties.__fields__.keys())
    table.add_column("key", justify="left", style="bold cyan")
    for key in fields:
        table.add_column(key, justify="center")

    for k, v in batch.items():
        props = infer_properties(v)
        table.add_row(k, *[str(getattr(props, f)) for f in fields])

    if console is None:
        console = rich.console.Console()

    console.print(table)
    return {}


def _iter_leaves(x: Iterable) -> Iterable:
    for i in x:
        if isinstance(i, (list, tuple, set)):
            yield from _iter_leaves(i)
        else:
            yield i
