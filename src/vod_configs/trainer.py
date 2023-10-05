import pathlib
import typing as typ

import pydantic
from typing_extensions import Self, Type
from vod_configs.utils.base import StrictModel
from vod_tools.misc.config import as_pyobj_validator
from vod_tools.misc.schedule import BaseSchedule, schedule_factory


class TrainerConfig(StrictModel):
    """Configures the Training step."""

    max_steps: int = 1_000_000
    val_check_interval: int = 500
    log_interval: int = 100
    gradient_clip_val: None | float = None
    period: int | list[int]
    parameters: dict[str, BaseSchedule] = {}
    n_max_eval: None | int = None
    checkpoint_path: None | str = None
    pbar_keys: list[str] = ["loss", "hitrate_3"]

    # validators
    _validate_update_freq = pydantic.field_validator("period", mode="before")(as_pyobj_validator)
    _validate_pbark_keys = pydantic.field_validator("pbar_keys", mode="before")(as_pyobj_validator)

    @pydantic.field_validator("parameters", mode="before")
    @classmethod
    def _validate_parameters(cls: Type[Self], x: None | dict[str, typ.Any]) -> dict[str, typ.Any]:
        if x is None:
            return {}

        params = {}
        for k, v in x.items():
            if isinstance(v, (float, int)):
                params[k] = schedule_factory(mode="constant", value=v)
            elif isinstance(v, BaseSchedule):
                params[k] = v
            else:
                params[k] = schedule_factory(**v)

        return params


class BenchmarkConfig(StrictModel):
    """Configures the batch size for the train, eval, and predict stages."""

    class Config:
        """Pydantic configuration."""

        extra = "forbid"
        frozen = False

    on_init: bool = False
    n_max_eval: None | int = None
    parameters: dict[str, float] = {}
    metrics: list[str] = ["ndcg", "mrr", "hitrate@01", "hitrate@03", "hitrate@10", "hitrate@30"]

    @pydantic.field_validator("metrics", mode="before")
    @classmethod
    def _validate_list(cls: Type[Self], v: None | list[str]) -> list[str]:
        if v is None:
            return []
        return [str(x) for x in v]


class BatchSizeConfig(StrictModel):
    """Configures the batch size for the train, eval, and predict stages."""

    effective: int = 32
    per_device: int = 64
    per_device_eval: int = 8
    per_device_predict: int = 512


class SysConfig(StrictModel):
    """Configures the system directories."""

    work_dir: pathlib.Path
    cache_dir: pathlib.Path
    nvme: pathlib.Path
    username: None | str = None
    hostname: None | str = None

    @pydantic.field_validator("work_dir", "cache_dir", "nvme", mode="before")
    @classmethod
    def _validate_paths(cls: Type[Self], v: str | pathlib.Path) -> pathlib.Path:
        return pathlib.Path(v).expanduser().resolve()
