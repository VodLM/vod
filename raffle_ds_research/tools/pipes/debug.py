from __future__ import annotations

import functools
import numbers
from numbers import Number
from typing import Any, Iterable, Optional

import loguru
import numpy as np
import pydantic
import rich
import rich.console
import rich.syntax
import rich.table
import rich.tree
import torch
import transformers
import yaml

_PPRINT_DISPLAY_PREC = 2


def _smart_str(x: Number) -> str:
    if isinstance(x, float):
        return f"{x:.{_PPRINT_DISPLAY_PREC}e}"
    if isinstance(x, int):
        return f"{x}"
    if isinstance(x, complex):
        return f"{x.real:.{_PPRINT_DISPLAY_PREC}e} + {x.imag:.{_PPRINT_DISPLAY_PREC}e}j"

    return str(x)


class Properties(pydantic.BaseModel):
    """Defines a set of displayable properties for a given object."""

    py_type: str
    dtype: Optional[str] = None
    shape: Optional[str] = None
    device: Optional[str] = None
    mean: Optional[str] = None
    min: Optional[str] = None
    max: Optional[str] = None
    n_nans: Optional[int] = None

    @pydantic.validator("py_type", pre=True)
    def _cast_py_type(cls, value: Any) -> str:  # noqa: ANN401
        if value is None:
            return "None"
        if isinstance(value, type):
            return value.__name__

        return type(value).__name__


@functools.singledispatch
def infer_properties(x: Any) -> Properties:  # noqa: ANN401
    """Base function for inferring properties of an object."""
    return Properties(py_type=type(x))


@infer_properties.register(torch.Tensor)
def _(x: torch.Tensor) -> Properties:
    """Infer properties of a torch tensor."""
    x_float = x.detach().float()
    return Properties(
        py_type=type(x),
        shape=str(list(x.shape)),
        dtype=str(x.dtype),
        device=str(x.device),
        mean=f"{_smart_str(x_float.mean().item())}",
        min=f"{_smart_str(x.min().item())}",
        max=f"{_smart_str(x.max().item())}",
        n_nans=int(torch.isnan(x).sum().item()),
    )


@infer_properties.register(np.ndarray)
def _(x: np.ndarray) -> Properties:
    """Infer properties of a numpy array."""
    x_float: np.ndarray = x.astype(np.float32)
    return Properties(
        py_type=type(x),
        shape=str(x.shape),
        dtype=f"np.{x.dtype}",
        device="-",
        mean=f"{_smart_str(x_float.mean())}",
        min=f"{_smart_str(np.min(x))}",
        max=f"{_smart_str(np.max(x))}",
        n_nans=int(np.isnan(x).sum()),
    )


@infer_properties.register(Number)
def _(x: Number) -> Properties:
    """Infer properties of a number."""
    return Properties(
        py_type=type(x),
        dtype="-",
        shape="-",
        device="-",
        min="-",
        max="-",
        mean=f"{_smart_str(x)}",
    )


@infer_properties.register(list)
@infer_properties.register(set)
@infer_properties.register(tuple)
def _(x: list | set | tuple) -> Properties:
    """Infer properties of a list, set or tuple."""
    try:
        arr = np.array(x)
        shape = str(arr.shape)
    except ValueError:
        shape = f"[{len(x)}, ?]"

    n_nans = sum(1 for y in _iter_leaves(x) if y is None)
    leaves_types = list({type(y) for y in _iter_leaves(x)})
    if all(issubclass(t, Number) for t in leaves_types):
        leaves_mean = np.mean(list(_iter_leaves(x)))
        leaves_min = min(_iter_leaves(x))
        leaves_max = max(_iter_leaves(x))
    else:
        leaves_mean = "-"
        leaves_min = "-"
        leaves_max = "-"

    def _format_type(x: type) -> str:
        if x == type(None):
            return "None"

        return str(x.__name__)

    leaves_types_ = [_format_type(t) for t in leaves_types]
    if len(leaves_types_) == 1:
        leaves_types_ = leaves_types_[0]

    return Properties(
        py_type=type(x),
        dtype=f"py.{leaves_types_}",
        shape=shape,
        min=leaves_min,
        max=leaves_max,
        device="-",
        mean=leaves_mean,
        n_nans=n_nans,
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
    """Format a dtype as a string."""
    if "." not in x:
        return f"[white]{x}[/]"
    type_, dtype_str = x.split(".")
    style = BASE_STYLES.get(type_, "")
    dtype_str = f"[{style}]{dtype_str}[/]"
    return f"[white]{type_}.[/white]{dtype_str}"


def _format_py_type(x: str) -> str:
    """Format a python type as a string."""
    style = {
        "list": BASE_STYLES["py"],
        "tuple": BASE_STYLES["py"],
        "set": BASE_STYLES["py"],
        "None": "bold red",
        "Tensor": BASE_STYLES["torch"],
        "ndarray": BASE_STYLES["np"],
    }.get(x, "white")
    return f"[{style}]{x}[/]"


def _format_device(device: str) -> str:
    """Format a device sas a string."""
    if device is None:
        return "-"
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

    return f"[{style}]{device}[/{style}]{idx}"


def _default_formatter(x: Any) -> str:  # noqa: ANN401
    """Default formatter."""
    x = str(x)
    if x.strip() == "-":
        return f"[white]{x}[/]"

    return x


_FORMATTERS = {
    "dtype": _format_dtype,
    "py_type": _format_py_type,
    "device": _format_device,
}


def _format_field(field_name: str, field_value: Any) -> str:  # noqa: ANN401
    """Apply a formatter to a field value based on its name."""
    formatter = _FORMATTERS.get(field_name, _default_formatter)
    return formatter(field_value)


def pprint_batch(
    batch: dict[str, Any],
    idx: Optional[list[int]] = None,  # noqa: ARG
    console: Optional[rich.console.Console] = None,
    header: Optional[str] = None,
    footer: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Pretty print a batch of data."""
    table = rich.table.Table(title=header, show_header=True, header_style="bold magenta")
    fields = list(Properties.__fields__.keys())
    table.add_column("key", justify="left", style="bold cyan")
    for key in fields:
        table.add_column(key, justify="center")

    for key, value in batch.items():
        try:
            props = infer_properties(value)
            attrs = {f: getattr(props, f) for f in fields}
            table.add_row(key, *[_format_field(k, v) for k, v in attrs.items()])
        except Exception as e:
            loguru.logger.warning(f"Error while inferring properties for `{key}={value}` : {e}")
            table.add_row(key, *["[red]ERROR[/red]"])
            raise e

    if console is None:
        console = rich.console.Console()

    console.print(table)
    if footer is not None:
        console.print(footer)
    return {}


def _iter_leaves(x: Iterable) -> Iterable:
    for i in x:
        if isinstance(i, (list, tuple, set)):
            yield from _iter_leaves(i)
        else:
            yield i


def _safe_yaml(section: str) -> str:
    """Escape special characters in a YAML section."""
    section = section.replace(": ", r"\:")
    section = section.encode("unicode_escape").decode("utf-8")
    return section


def pprint_retrieval_batch(
    batch: dict[str, Any],
    idx: Optional[list[int]] = None,  # noqa: ARG
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
    header: str = "Supervised retrieval batch",
    max_sections: Optional[int] = None,
    console: Optional[rich.console.Console] = None,
    **kwargs: Any,
) -> dict:
    """Pretty print a batch of data for supervised retrieval."""
    if console is None:
        console = rich.console.Console()

    def _format(x: Any | np.ndarray | torch.Tensor) -> numbers.Number | list[numbers.Number]:
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            x = x.item()
        elif isinstance(x, torch.Tensor) and x.numel() > 1:
            x = x.tolist()
        elif isinstance(x, np.ndarray) and x.size == 1:
            x = x.item()

        return x

    tree = rich.tree.Tree(header, guide_style="dim")
    question_keys = ["id", "section_ids", "answer_id", "kb_id", "language", "group_hash"]
    question_keys = [f"question.{key}" for key in question_keys]
    section_keys = ["id", "answer_id", "kb_id", "score", "label", "language", "group_hash"]
    section_keys = [f"section.{key}" for key in section_keys]

    # get the data
    question_input_ids = batch["question.input_ids"]
    section_input_ids = batch["section.input_ids"]

    for i, q_ids in enumerate(question_input_ids):
        question = tokenizer.decode(q_ids, **kwargs)
        question_data = {
            **{key: str(_format(batch[key][i])) for key in question_keys if key in batch},
            "question.content": _safe_yaml(question),
        }
        question_data_str = yaml.dump(question_data, sort_keys=False)
        question_node = rich.syntax.Syntax(question_data_str, "yaml", indent_guides=False, word_wrap=True)
        question_tree = rich.tree.Tree(question_node, guide_style="dim")
        for j in range(len(section_input_ids[i])):
            section = tokenizer.decode(section_input_ids[i][j], **kwargs)
            section_data = {
                **{str(key): _format(batch[key][i][j]) for key in section_keys if key in batch},  # type: ignore
                "section.content": _safe_yaml(section),
            }
            section_data_str = yaml.dump(section_data, sort_keys=False)
            section_node = rich.syntax.Syntax(section_data_str, "yaml", indent_guides=False, word_wrap=True)
            node_style = "bold cyan" if section_data.get("section.label", False) else "white"

            question_tree.add(section_node, style=node_style)
            if max_sections is not None and j >= max_sections:
                break

        tree.add(question_tree)

    console.print(tree)

    return {}
