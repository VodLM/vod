import typing as typ
from collections.abc import Mapping

import omegaconf

T = typ.TypeVar("T")


def maybe_cast_omegaconf(
    x: T | omegaconf.DictConfig | omegaconf.ListConfig,
    **kwargs: typ.Any,
) -> T | dict | None | str | typ.Any:  # noqa: ANN401
    """Cast an OmegaConf object to a dict."""
    if isinstance(x, (omegaconf.DictConfig)):
        return omegaconf.OmegaConf.to_container(x, **kwargs)

    if isinstance(x, (omegaconf.ListConfig)):
        return omegaconf.OmegaConf.to_container(x, **kwargs)

    return x


def as_pyobj_validator(
    cls: typ.Any,  # noqa: ANN401, ARG001
    x: T | omegaconf.DictConfig | omegaconf.ListConfig,
) -> T | dict | None | str | typ.Any:  # noqa: ANN401
    """Pydantic validator to cast an OmegaConf object to a dict/list."""
    return maybe_cast_omegaconf(x, resolve=True)


def flatten_dict(node: Mapping[typ.Any, dict | typ.Any], sep: str = ".") -> dict[str, typ.Any]:
    """Flatten a nested dictionary. Keys are joined with `sep`.

    Example:
    ```
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}
    ```
    """
    output = {}
    for k, v in node.items():
        if isinstance(v, Mapping):
            for k2, v2 in flatten_dict(v, sep=sep).items():
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
