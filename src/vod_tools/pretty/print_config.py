import contextlib
from copy import copy
from numbers import Number

import rich
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.syntax import Syntax
from rich.tree import Tree


def pprint_config(
    config: DictConfig,
    fields: None | list[str] = None,
    resolve: bool = True,
    exclude: None | list[str] = None,
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

    fields_list: list[str] = fields or list(config.keys())  # type: ignore
    fields_list = list(filter(lambda x: x not in exclude, fields_list))

    with open_dict(config):
        base_config = {}
        for field in copy(fields_list):
            field_value = config.get(field)
            if field_value is None or isinstance(field_value, (bool, str, Number)):
                base_config[field] = copy(field_value)
                fields_list.remove(field)
        config["__root__"] = base_config
    fields_list = ["__root__"] + fields_list

    for field in fields_list:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        if isinstance(config_section, DictConfig):
            with contextlib.suppress(Exception):
                branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        else:
            branch_content = str(config_section)

        branch.add(Syntax(branch_content, "yaml", indent_guides=True, word_wrap=True))

    rich.print(tree)
