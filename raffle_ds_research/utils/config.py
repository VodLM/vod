from __future__ import annotations

from typing import Any

import omegaconf
from omegaconf import DictConfig, OmegaConf


def flatten_dict(node: dict[str, dict | Any], sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary. Keys are joined with `sep`.

    Example:
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}

    """
    output = {}
    for k, v in node.items():
        if isinstance(v, dict):
            for k2, v2 in flatten_dict(v).items():
                output[f"{k}{sep}{k2}"] = v2
        else:
            output[k] = v
    return output


def config_to_flat_dict(config: DictConfig, resolve: bool = True, sep: str = ".") -> dict[str, str]:
    """Convert a config to a flat dictionary."""
    if isinstance(config, omegaconf.DictConfig):
        config = OmegaConf.to_container(config, resolve=resolve)
    flat_config = flatten_dict(config, sep=sep)
    flat_config = {k: v for k, v in flat_config.items()}
    return flat_config
