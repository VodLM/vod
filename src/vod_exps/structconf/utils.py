import copy
import itertools
import re
import typing as typ
from collections.abc import Mapping as MappingABC

import omegaconf as omg

Cv = typ.TypeVar("Cv", bound=str | dict | list)

_CONFIG_EXPAND_KEY = "__vars__"


def resolve_configs_list(x: list[typ.Mapping[str, typ.Any]]) -> list[dict[str, typ.Any]]:
    """Dynamically expand a list of configurations whenever a `__vars__` key is found."""
    expanded_x = []
    for y in x:
        if not isinstance(y, dict):
            expanded_x.append(y)
            continue
        variables = y.pop(_CONFIG_EXPAND_KEY, None)  # type: ignore
        if variables is None:
            expanded_x.append(y)
            continue

        # Take the combinations of the variables
        def _sub(v: Cv, target: str, value: typ.Any) -> Cv:  # noqa: ANN401
            if isinstance(v, str):
                # replace `{target}` with `value`
                return re.sub(rf"\{{\s*{target}\s*\}}", str(value), v)
            if isinstance(v, dict):
                return {k: _sub(v, target, value) for k, v in v.items()}  # type: ignore
            if isinstance(v, list):
                return [_sub(v, target, value) for v in v]  # type: ignore
            return v

        keys = list(variables.keys())
        values = list(variables.values())
        for comb in itertools.product(*values):
            new_y = copy.deepcopy(y)
            for pat, val in zip(keys, comb):
                new_y = {k: _sub(v, pat, val) for k, v in new_y.items()}
            expanded_x.append(new_y)

    return expanded_x


def to_dicts_list(
    x: typ.Mapping[str, typ.Any] | list[typ.Mapping[str, typ.Any]],
) -> list[typ.Mapping[str, typ.Any]]:
    """Convert an omegaconf object to a list of dictionaries."""
    if isinstance(x, (omg.DictConfig, omg.ListConfig)):
        x = omg.OmegaConf.to_container(x, resolve=True)  # type: ignore

    if isinstance(x, MappingABC):
        x = [x]

    return [dict(y) for y in x]
