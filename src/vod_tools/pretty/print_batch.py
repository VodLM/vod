import copy
import functools
import math
import numbers
import pathlib
import typing as typ
from numbers import Number

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
from typing_extensions import Self, Type
from vod_tools.misc.config import flatten_dict

_PPRINT_PREC = 2


def _smart_str(x: Number) -> str:
    if isinstance(x, float):
        return f"{x:.{_PPRINT_PREC}e}"
    if isinstance(x, int):
        return f"{x}"
    if isinstance(x, complex):
        return f"{x.real:.{_PPRINT_PREC}e} + {x.imag:.{_PPRINT_PREC}e}j"

    return str(x)


class Properties(pydantic.BaseModel):
    """Defines a set of displayable properties for a given object."""

    py_type: str
    dtype: None | str = None
    shape: None | str = None
    device: None | str = None
    mean: None | str = None
    min: None | str = None
    max: None | str = None
    n_nans: None | int = None

    @pydantic.field_validator("py_type", mode="before")
    @classmethod
    def _cast_py_type(cls: Type[Self], value: typ.Any) -> str:  # noqa: ANN401
        if value is None:
            return "None"
        if isinstance(value, type):
            return value.__name__

        return type(value).__name__


@functools.singledispatch
def infer_properties(x: typ.Any) -> Properties:  # noqa: ANN401
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


@infer_properties.register(dict)
def _(x: dict) -> Properties:
    """Infer properties of a number."""
    return Properties(
        py_type=type(x),
        dtype="-",
        shape="-",
        device="-",
        min="-",
        max="-",
        mean="-",
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
    try:
        leaves_mean = np.mean(list(_iter_leaves(x)))
        leaves_min = min(_iter_leaves(x))
        leaves_max = max(_iter_leaves(x))
    except Exception:
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
        "dict": BASE_STYLES["py"],
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


def _default_formatter(x: typ.Any) -> str:  # noqa: ANN401
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


def _format_field(field_name: str, field_value: typ.Any) -> str:  # noqa: ANN401
    """Apply a formatter to a field value based on its name."""
    formatter = _FORMATTERS.get(field_name, _default_formatter)
    return formatter(field_value)


def pprint_batch(
    batch: dict[str, typ.Any],
    idx: None | list[int] = None,  # noqa: ARG - used here to enable compatibility with `datasets.map()`
    console: None | rich.console.Console = None,
    header: None | str = None,
    footer: None | str = None,
    **kwargs: typ.Any,
) -> dict:
    """Pretty print a batch of data."""
    table = rich.table.Table(title=header, show_header=True, header_style="bold magenta")
    fields = list(Properties.__fields__.keys())
    table.add_column("key", justify="left", style="bold cyan")
    for key in fields:
        table.add_column(key, justify="center")

    # Convert dict to flat dict
    flat_batch = flatten_dict(batch)
    for key, value in flat_batch.items():
        try:
            props = infer_properties(value)
            attrs = {f: getattr(props, f) for f in fields}
            table.add_row(key, *[_format_field(k, v) for k, v in attrs.items()])
        except Exception as e:
            loguru.logger.warning(f"Error while inferring properties for `{key}={value}` : {e}")
            table.add_row(key, *["[red]ERROR[/red]"])

    if console is None:
        console = rich.console.Console()

    console.print(table)
    if footer is not None:
        console.print(footer)
    return {}


def _iter_leaves(x: typ.Iterable) -> typ.Iterable:
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


def pprint_retrieval_batch(  # noqa: C901, PLR0915, PLR0912
    batch: dict[str, typ.Any],
    idx: None | list[int] = None,  # noqa: ARG
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
    header: str = "Supervised retrieval batch",
    max_questions: None | int = None,
    max_sections: None | int = None,
    console: None | rich.console.Console = None,
    output_file: None | str | pathlib.Path = None,
    footer: str | bool = True,
    **kwargs: typ.Any,
) -> dict:
    """Pretty print a batch of data for supervised retrieval."""
    if console is None:
        console = rich.console.Console()

    def _format(x: typ.Any | np.ndarray | torch.Tensor) -> numbers.Number | list[numbers.Number]:
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            x = x.item()
        elif isinstance(x, torch.Tensor) and x.numel() > 1:
            x = x.tolist()
        elif isinstance(x, np.ndarray) and x.size == 1:
            x = x.item()

        return x

    tree = rich.tree.Tree(header, guide_style="dim")
    query_keys = ["id", "retrieval_ids", "subset_ids", "language"]
    query_keys = [f"query__{key}" for key in query_keys]
    section_keys = ["id", "subset_id", "score", "log_weight", "label", "language"]
    section_keys = [f"section__{key}" for key in section_keys]
    need_expansion = [
        "section__input_ids",
        "section__attention_mask",
        "section__token_type_ids",
        "section__idx",
        "section__id",
        "section__language",
        "section__subset_id",
    ]

    # Fetch the querys
    batch = copy.deepcopy(batch)  # noqa: F821
    query_input_ids = batch["query__input_ids"]
    batch_size = len(query_input_ids)

    # Fetch and expand section attributes if needed
    if batch["section__input_ids"].ndim == 2:  # noqa: PLR2004
        # Sections are flatten, expand them
        for k, v in batch.items():
            if k in need_expansion:
                if isinstance(v, torch.Tensor):
                    batch[k] = v[None, :].expand(batch_size, *(-1 for _ in v.shape))
                elif isinstance(v, np.ndarray):
                    batch[k] = v[None, :].repeat(batch_size, axis=0)
                elif isinstance(v, list):
                    batch[k] = [v for _ in range(batch_size)]
                else:
                    raise TypeError(f"Cannot expand {k} of type {type(v)}")
    elif batch["section__input_ids"].ndim == 3:  # noqa: PLR2004
        # We have K sections per queries, that's the default case
        pass
    else:
        raise ValueError(
            f"Section input ids should be a 2D or 3D tensor. Found shape: `{batch['section__input_ids'].shape}`"
        )

    section_input_ids = batch["section__input_ids"]
    for i, q_ids in enumerate(query_input_ids):
        if max_questions is not None and i >= max_questions:
            break
        query = tokenizer.decode(q_ids, **kwargs)
        query_data = {
            **{key: str(_format(batch[key][i])) for key in query_keys if key in batch},
            "query__content": _safe_yaml(query),
        }
        query_data_str = yaml.dump(query_data, sort_keys=False)
        query_node = rich.syntax.Syntax(query_data_str, "yaml", indent_guides=False, word_wrap=True)
        query_tree = rich.tree.Tree(query_node, guide_style="dim")

        # sort the querys by positive label (first), then higher score (second)
        _indices_i = range(len(batch["section__label"][i]))
        _labels_i = [float(x > 0) for x in batch["section__label"][i]]
        sort_scores = _cleanup_scores(batch["section__score"][i])
        section_sort_ids = [
            i for i, _, _ in sorted(zip(_indices_i, _labels_i, sort_scores), key=lambda x: (x[1], x[2]), reverse=True)
        ]

        for count, j in enumerate(section_sort_ids):
            section = tokenizer.decode(section_input_ids[i][j], **kwargs)
            section_data = {
                **{str(key): _format(batch[key][i][j]) for key in section_keys if key in batch},  # type: ignore
                "section__content": _safe_yaml(section),
            }
            section_data_str = yaml.dump(section_data, sort_keys=False)
            section_node = rich.syntax.Syntax(section_data_str, "yaml", indent_guides=False, word_wrap=True)
            node_style = "bold cyan" if section_data.get("section__label", False) else "white"

            query_tree.add(section_node, style=node_style)
            if max_sections is not None and count >= max_sections:
                break

        tree.add(query_tree)

    if output_file is not None:
        with pathlib.Path(output_file).open("w") as f:
            rich.print(tree, file=f)

    console.print(tree)
    if isinstance(footer, str):
        console.print(footer)
    elif footer:  # print a line separator
        console_width = console.size.width
        console.print("-" * console_width)

    return {}


def _cleanup_scores(sort_scores: list[float]) -> list[float]:
    non_nan_scores = [x for x in sort_scores if not math.isnan(x)]
    min_score = min(non_nan_scores) if len(non_nan_scores) > 0 else 0.0
    sort_scores = [min_score - 1 if math.isnan(x) else x for x in sort_scores]
    return sort_scores
