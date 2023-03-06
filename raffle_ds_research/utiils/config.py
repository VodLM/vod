from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def flatten_dict(node: dict[str, dict | Any]) -> dict[str, Any]:
    output = {}
    for k, v in node.items():
        if isinstance(v, dict):
            for k2, v2 in flatten_dict(v).items():
                output[f"{k}.{k2}"] = str(v2)
        else:
            output[k] = str(v)
    return output


def config_to_flat_dict(config: DictConfig):
    config = OmegaConf.to_container(config, resolve=True)
    return flatten_dict(config)
