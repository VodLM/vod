from __future__ import annotations

from copy import copy
from numbers import Number
from typing import List, Optional

import rich
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.syntax import Syntax
from rich.tree import Tree


def human_format_nb(num: int | float, precision: int = 2) -> str:
    """Converts a number to a human-readable format."""
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
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
            if isinstance(config.get(field), (bool, str, Number)):
                base_config[field] = copy(config.get(field))
                fields.remove(field)
        config["__root__"] = base_config
    fields = ["__root__"] + fields

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        if isinstance(config_section, DictConfig):
            try:
                branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
            except Exception:
                pass
        else:
            branch_content = str(config_section)

        branch.add(Syntax(branch_content, "yaml", indent_guides=True, word_wrap=True))

    rich.print(tree)


def repr_tensor(x: torch.Tensor) -> str:
    """Return a string representation of a tensor."""
    return f"Tensor(shape={x.shape}, dtype={x.dtype}, device={x.device})"
