from __future__ import annotations

import functools
from numbers import Number
from typing import Any, Iterable, Optional

import loguru
import numpy as np
import pydantic
import rich
import rich.console
import rich.table
import torch
import transformers
import yaml

_PPRINT_DISPLAY_PREC = 2


def _smart_str(x: Number) -> str:
    if isinstance(x, float):
        return f"{x:.{_PPRINT_DISPLAY_PREC}e}"
    elif isinstance(x, int):
        return f"{x}"
    elif isinstance(x, complex):
        return f"{x.real:.{_PPRINT_DISPLAY_PREC}e} + {x.imag:.{_PPRINT_DISPLAY_PREC}e}j"
    else:
        return str(x)


class Properties(pydantic.BaseModel):
    py_type: str
    dtype: Optional[str] = None
    shape: Optional[str] = None
    device: Optional[str] = None
    mean: Optional[str] = None
    min: Optional[str] = None
    max: Optional[str] = None

    @pydantic.validator("py_type", pre=True)
    def _cast_py_type(cls, v: Any) -> str:
        if v is None:
            return "None"
        elif isinstance(v, type):
            return v.__name__
        else:
            return type(v).__name__


@functools.singledispatch
def infer_properties(x: Any) -> Properties:
    return Properties(py_type=type(x))


@infer_properties.register(torch.Tensor)
def _(x: torch.Tensor) -> Properties:
    xf = x.detach().float()
    return Properties(
        py_type=type(x),
        shape=str(list(x.shape)),
        dtype=str(x.dtype),
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
        dtype=f"np.{x.dtype}",
        device="-",
        mean=f"{_smart_str(xf.mean())}",
        min=f"{_smart_str(np.min(x))}",
        max=f"{_smart_str(np.max(x))}",
    )


@infer_properties.register(Number)
def _(x: Number) -> Properties:
    return Properties(
        py_type=type(x),
        dtype="-",
        mean=f"{_smart_str(x)}",
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
        leaves_mean = np.mean([y for y in _iter_leaves(x)])
        leaves_min = min(_iter_leaves(x))
        leaves_max = max(_iter_leaves(x))
    else:
        leaves_mean = "-"
        leaves_min = "-"
        leaves_max = "-"

    def _format_type(x: type) -> str:
        if x == type(None):
            return "None"
        else:
            return x.__name__

    leaves_types = [_format_type(t) for t in leaves_types]
    if len(leaves_types) == 1:
        leaves_types = leaves_types[0]

    return Properties(
        py_type=type(x),
        dtype=f"py.{leaves_types}",
        shape=shape,
        min=leaves_min,
        max=leaves_max,
        device="-",
        mean=leaves_mean,
    )


BASE_STYLES = {
    "torch": "bold cyan",
    "np": "bold green",
    "py": "bold yellow",
}

BASE_DEVICE_STYLES = {
    "cpu": "bold yellow",
    "cuda": "bold magenta",
}


def _format_dtype(x: str) -> str:
    type_, dtype_str = x.split(".")
    style = BASE_STYLES.get(type_, "")
    dtype_str = f"[{style}]{dtype_str}[/]"
    return f"[white]{type_}.[/white]{dtype_str}"


def _format_py_type(x: str) -> str:
    style = {
        "list": BASE_STYLES["py"],
        "tuple": BASE_STYLES["py"],
        "set": BASE_STYLES["py"],
        "None": "bold red",
        "Tensor": BASE_STYLES["torch"],
        "ndarray": BASE_STYLES["np"],
    }.get(x, "")
    return f"[{style}]{x}[/]"


def _format_device(device: str) -> str:
    if device.strip() == "-":
        return _default_formatter(device)
    if ":" in device:
        device, idx = device.split(":")
        idx = f"[bold white]:{idx}[/bold white]"
    else:
        idx = ""
    style = BASE_DEVICE_STYLES.get(device, None)
    if style is None:
        return f"{device}{idx}"
    else:
        return f"[{style}]{device}[/{style}]{idx}"


def _default_formatter(x: Any) -> str:
    x = str(x)
    if x.strip() == "-":
        return f"[white]{x}[/]"
    else:
        return x


_FORMATTERS = {
    "dtype": _format_dtype,
    "py_type": _format_py_type,
    "device": _format_device,
}


def _format_field(field_name, field_value):
    formatter = _FORMATTERS.get(field_name, _default_formatter)
    return formatter(field_value)


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
        try:
            props = infer_properties(v)
            attrs = {f: getattr(props, f) for f in fields}
            table.add_row(k, *[_format_field(k, v) for k, v in attrs.items()])
        except Exception as e:
            loguru.logger.warning(f"Error while inferring properties for `{k}={v}` : {e}")
            table.add_row(k, *[f"[red]ERROR[/red]"])
            raise e

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


def pprint_supervised_retrieval_batch(
    batch: dict[str, Any],
    idx: Optional[list[int]] = None,
    *,
    tokenizer: transformers.PreTrainedTokenizer,
    header="Supervised retrieval batch",
    console: Optional[rich.console.Console] = None,
    **kwargs,
) -> dict:
    import rich.syntax
    import rich.tree

    if console is None:
        console = rich.console.Console()

    def _format(x: Any):
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            x = x.item()
        elif isinstance(x, np.ndarray) and x.size == 1:
            x = x.item()

        return x

    tree = rich.tree.Tree(header, guide_style="dim")
    question_keys = ["id", "section_id", "answer_id", "kb_id"]
    question_keys = [f"question.{key}" for key in question_keys]
    section_keys = ["id", "answer_id", "kb_id", "score", "label"]
    section_keys = [f"section.{key}" for key in section_keys]

    # get the data
    question_input_ids = batch["question.input_ids"]
    section_input_ids = batch["section.input_ids"]

    for i in range(len(question_input_ids)):
        question = tokenizer.decode(question_input_ids[i], **kwargs)
        question_data = {
            **{key: _format(batch[key][i]) for key in question_keys if key in batch},
            "question.content": question,
        }
        question_data_str = yaml.dump(question_data, sort_keys=False)
        question_node = rich.syntax.Syntax(question_data_str, "yaml", indent_guides=False, word_wrap=True)
        question_tree = rich.tree.Tree(question_node, guide_style="dim")
        for j in range(len(section_input_ids[i])):
            section = tokenizer.decode(section_input_ids[i][j], **kwargs)
            section_data = {
                **{key: _format(batch[key][i][j]) for key in section_keys if key in batch},
                "section.content": section,
            }
            section_data_str = yaml.dump(section_data, sort_keys=False)
            section_node = rich.syntax.Syntax(section_data_str, "yaml", indent_guides=False, word_wrap=True)
            question_tree.add(section_node)

        tree.add(question_tree)

    console.print(tree)

    return {}
