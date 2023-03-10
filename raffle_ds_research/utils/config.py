from __future__ import annotations

from typing import Any

import rich
from omegaconf import DictConfig, OmegaConf


def flatten_dict(node: dict[str, dict | Any]) -> dict[str, Any]:
    output = {}
    for k, v in node.items():
        if isinstance(v, dict):
            for k2, v2 in flatten_dict(v).items():
                output[f"{k}.{k2}"] = v2
        else:
            output[k] = v
    return output


def config_to_flat_dict(config: DictConfig, resolve: bool = True) -> dict[str, str]:
    config = OmegaConf.to_container(config, resolve=resolve)
    flat_config = flatten_dict(config)
    flat_config = {k: str(v) for k, v in flat_config.items()}
    return flat_config
