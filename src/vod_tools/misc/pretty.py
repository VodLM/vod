import contextlib
import dataclasses
from copy import copy
from numbers import Number
from typing import Any, List, Literal, Optional

import rich
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.syntax import Syntax
from rich.tree import Tree


def human_format_bytes(x: int, unit: Literal["KB", "MB", "GB"]) -> str:
    """Format bytes to a human readable format."""
    divider = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }
    return f"{x / divider[unit]:.2f} {unit}"


def human_format_nb(num: int | float, precision: int = 2, base: float = 1000.0) -> str:
    """Converts a number to a human-readable format."""
    magnitude = 0
    while abs(num) >= base:
        magnitude += 1
        num /= base
    # add more suffixes if you need them
    q = ["", "K", "M", "G", "T", "P"][magnitude]
    return f"{num:.{precision}f}{q}"


def print_config(
    config: DictConfig,
    fields: Optional[List[str]] = None,
    resolve: bool = True,
    exclude: Optional[List[str]] = None,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        exclude (List[str], optional): fields to exclude.
    """
    config = copy(config)

    style = "dim"
    tree = Tree(":gear: CONFIG", style=style, guide_style=style)
    if exclude is None:
        exclude = []

    fields = fields or list(config.keys())
    fields = list(filter(lambda x: x not in exclude, fields))

    with open_dict(config):
        base_config = {}
        for field in copy(fields):
            field_value = config.get(field)
            if field_value is None or isinstance(field_value, (bool, str, Number)):
                base_config[field] = copy(field_value)
                fields.remove(field)
        config["__root__"] = base_config
    fields = ["__root__"] + fields

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        if isinstance(config_section, DictConfig):
            with contextlib.suppress(Exception):
                branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        else:
            branch_content = str(config_section)

        branch.add(Syntax(branch_content, "yaml", indent_guides=True, word_wrap=True))

    rich.print(tree)


def repr_tensor(x: torch.Tensor) -> str:
    """Return a string representation of a tensor."""
    return f"Tensor(shape={x.shape}, dtype={x.dtype}, device={x.device})"


@dataclasses.dataclass
class TableRow:
    """A row in the table."""

    key: str
    group: str
    value: float


def print_metric_groups(metrics: dict[str, Any], header: None | str = None) -> None:
    """Nicely print a dictionary of metrics using `rich.table.Table`."""
    console = rich.console.Console()
    table = rich.table.Table(title=header or "Static Validation")
    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("group", justify="left", style="green")
    table.add_column("Value", justify="right", style="magenta")

    def _make_row(key: str, value: Any) -> TableRow:  # noqa: ANN401
        *group, metric = key.split("/")
        group = "/".join(group)
        return TableRow(key=metric, group=group, value=float(value))

    metrics = [_make_row(k, v) for k, v in metrics.items()]
    metrics = sorted(metrics, key=lambda x: (x.key, x.group))
    prev_key = None
    for row in metrics:
        if prev_key is None:
            display_key = row.key
        elif row.key != prev_key:
            table.add_section()
            display_key = row.key
        else:
            display_key = ""
        table.add_row(display_key, row.group, f"{row.value:.2%}")
        prev_key = row.key
    console.print(table)
