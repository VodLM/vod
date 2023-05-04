from __future__ import annotations

from typing import Any, TypeVar

import omegaconf

T = TypeVar("T")


def maybe_cast_omegaconf(
    x: T | omegaconf.DictConfig | omegaconf.ListConfig,
    **kwargs: Any,
) -> T | dict | None | str | Any:
    """Cast an OmegaConf object to a dict."""
    if isinstance(x, (omegaconf.DictConfig)):
        return omegaconf.OmegaConf.to_container(x, **kwargs)

    if isinstance(x, (omegaconf.ListConfig)):
        return omegaconf.OmegaConf.to_container(x, **kwargs)

    return x


def as_pyobj_validator(
    cls: Any, x: T | omegaconf.DictConfig | omegaconf.ListConfig  # noqa: ARG, ANN
) -> T | dict | None | str | Any:
    """Pydantic validator to cast an OmegaConf object to a dict/list."""
    return maybe_cast_omegaconf(x, resolve=True)


def flatten_dict(node: dict[Any, dict | Any], sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary. Keys are joined with `sep`.

    Example:
    ```
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}
    ```
    """
    output = {}
    for k, v in node.items():
        if isinstance(v, dict):
            for k2, v2 in flatten_dict(v).items():
                output[f"{k}{sep}{k2}"] = v2
        else:
            output[k] = v
    return output


def config_to_flat_dict(config: dict | omegaconf.DictConfig, resolve: bool = True, sep: str = ".") -> dict[str, str]:
    """Convert a config to a flat dictionary."""
    py_config: dict = maybe_cast_omegaconf(config, resolve=resolve)  # type: ignore
    flat_config = flatten_dict(py_config, sep=sep)
    flat_config = dict(flat_config.items())
    return flat_config
