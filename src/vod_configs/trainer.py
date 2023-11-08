import pathlib
import typing as typ

import pydantic
from typing_extensions import Self, Type
from vod_configs.utils.base import StrictModel
from vod_tools.misc.config import as_pyobj_validator

from .utils.schedule import ParameterSchedule

BenchmarkStrategy = typ.Literal["retrieval", "generative", "rag"]  # Benchmarking strategies


class TrainerConfig(StrictModel):
    """Configures the Training step."""

    max_steps: int = 1_000_000
    val_check_interval: int = 500
    log_interval: int = 100
    gradient_clip_val: None | float = None
    period: int | list[int]
    accumulate_grad_batches: int = 1
    parameters: dict[str, ParameterSchedule] = {}
    n_max_eval: None | int = None
    checkpoint_path: None | str = None
    metrics: list[str] = ["kldiv", "ndcg_10", "mrr_10", "hitrate_01", "hitrate_03"]
    pbar_keys: list[str] = ["kldiv", "ndcg_10"]

    # validators
    _validate_update_freq = pydantic.field_validator("period", mode="before")(as_pyobj_validator)
    _validate_metrics = pydantic.field_validator("metrics", mode="before")(as_pyobj_validator)
    _validate_pbark_keys = pydantic.field_validator("pbar_keys", mode="before")(as_pyobj_validator)

    @pydantic.field_validator("parameters", mode="before")
    @classmethod
    def _validate_parameters(cls: Type[Self], x: None | typ.Mapping[str, typ.Any]) -> dict[str, ParameterSchedule]:
        if x is None:
            return {}
        return {k: ParameterSchedule.parse(v) for k, v in x.items()}


class BenchmarkConfig(StrictModel):
    """Configures the batch size for the train, eval, and predict stages."""

    class Config:
        """Pydantic configuration."""

        extra = "forbid"
        frozen = False

    on_init: bool = False
    n_max_eval: None | int = None
    parameters: dict[str, float] = {}
    metrics: list[str] = ["ndcg_10", "mrr_10", "hitrate_01", "hitrate_03", "kldiv"]
    serve_search_on_gpu: bool = False
    strategy: BenchmarkStrategy = "retrieval"

    @pydantic.field_validator("metrics", mode="before")
    @classmethod
    def _validate_list(cls: Type[Self], v: None | typ.Iterable[str]) -> list[str]:
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
