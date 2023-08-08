from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Optional, Union

import hydra
import omegaconf
import pydantic
from omegaconf import DictConfig, ListConfig
from typing_extensions import Self, Type
from vod_configs.py.models import TokenizerConfig
from vod_configs.py.utils import StrictModel
from vod_tools.misc.config import as_pyobj_validator
from vod_tools.misc.schedule import BaseSchedule, schedule_factory

from .dataloaders import BaseCollateConfig, DataLoaderConfig, RetrievalCollateConfig, SamplerFactoryConfig
from .datasets import DatasetsConfig


class TuningConfig(StrictModel):
    """Configures the batch size for the train, eval, and predict stages."""

    steps: int = 1000
    batch_size: int = 100
    learning_rate: float = 1e-3
    collate_overrides: dict[str, Any] = {}

    @pydantic.validator("collate_overrides", pre=True)
    def _validate_collate_overrides(cls, v: None | dict[str, Any]) -> dict[str, Any]:
        if v is None:
            return {}

        if isinstance(v, omegaconf.DictConfig):
            return omegaconf.OmegaConf.to_container(v, resolve=True)  # type: ignore

        return v


class BenchmarkConfig(StrictModel):
    """Configures the batch size for the train, eval, and predict stages."""

    class Config:
        """Pydantic configuration."""

        extra = "forbid"
        allow_mutation = False

    on_init: bool = False
    n_max_eval: Optional[int] = None
    tuning: Optional[TuningConfig] = None
    parameters: dict[str, float] = {}
    metrics: list[str] = ["ndcg", "mrr", "hitrate@01", "hitrate@03", "hitrate@10", "hitrate@30"]
    search: dict[str, Any] = {}

    # Validators
    _validate_metrics = pydantic.validator("metrics", allow_reuse=True, pre=True)(as_pyobj_validator)

    @pydantic.validator("search", pre=True)
    def _validate_searchs(cls, v: None | dict[str, Any]) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, omegaconf.DictConfig):
            return omegaconf.OmegaConf.to_container(v)  # type: ignore

        return v

    @classmethod
    def parse(cls: Type[Self], obj: dict | Self) -> Self:
        """Parse a benchmark config."""
        if isinstance(obj, cls):
            return obj

        return cls(**obj)  # type: ignore


class TrainerConfig(StrictModel):
    """Configures the Training step."""

    max_steps: int = 1_000_000
    val_check_interval: int = 500
    log_interval: int = 100
    gradient_clip_val: Optional[float] = None
    period: Union[int, list[int]]
    parameters: dict[str, BaseSchedule] = {}
    n_max_eval: Optional[int] = None
    checkpoint_path: Optional[str] = None
    pbar_keys: list[str] = ["loss", "hitrate_3"]

    # validators
    _validate_update_freq = pydantic.validator("period", allow_reuse=True, pre=True)(as_pyobj_validator)
    _validate_pbark_keys = pydantic.validator("pbar_keys", allow_reuse=True, pre=True)(as_pyobj_validator)

    @pydantic.validator("parameters", pre=True)
    def _validate_parameters(cls, x: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if x is None:
            return {}

        params = {}
        for k, v in x.items():
            if isinstance(v, (dict, omegaconf.DictConfig)):
                params[k] = schedule_factory(**v)
            elif isinstance(v, (float, int)):
                params[k] = schedule_factory(mode="constant", value=v)
            else:
                params[k] = v

        return params

    @classmethod
    def parse(cls: Type[Self], obj: dict | Self) -> Self:
        """Parse a benchmark config."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, omegaconf.DictConfig):
            return hydra.utils.instantiate(obj)

        return cls(**obj)  # type: ignore


class BatchSizeConfig(StrictModel):
    """Configures the batch size for the train, eval, and predict stages."""

    effective: int = 32
    per_device: int = 64
    per_device_eval: int = 8
    per_device_predict: int = 512

    @classmethod
    def parse(cls: Type[Self], obj: dict | Self) -> Self:
        """Parse a benchmark config."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, omegaconf.DictConfig):
            return hydra.utils.instantiate(obj)

        return cls(**obj)  # type: ignore


@dataclasses.dataclass
class DataLoaderConfigs:
    """Configures the `torch.utils.data.Dataloader` for the train, eval, and predict stages."""

    train: DataLoaderConfig
    eval: DataLoaderConfig
    benchmark: DataLoaderConfig
    predict: DataLoaderConfig

    @classmethod
    def parse(cls: Type[Self], config: DictConfig) -> Self:
        """Parse an omegaconf config into a DataLoaderConfigs instance."""
        return cls(
            train=DataLoaderConfig(**config.train),
            eval=DataLoaderConfig(**config.eval),
            benchmark=DataLoaderConfig(**config.benchmark),
            predict=DataLoaderConfig(**config.predict),
        )


@dataclasses.dataclass
class CollateConfigs:
    """Configures the collate functions for the train, eval, and static data loaders."""

    train: RetrievalCollateConfig
    benchmark: RetrievalCollateConfig
    predict: BaseCollateConfig

    @classmethod
    def parse(cls: Type[Self], config: DictConfig) -> Self:
        """Parse an omegaconf config into a CollateConfigs instance."""
        return cls(
            train=RetrievalCollateConfig(**config.train),
            benchmark=RetrievalCollateConfig(**config.benchmark),
            predict=BaseCollateConfig(**config.predict),
        )


@dataclasses.dataclass
class SysConfig:
    """Configures the system directories."""

    raffle_path: pathlib.Path
    work_dir: pathlib.Path
    cache_dir: pathlib.Path

    @classmethod
    def parse(cls: Type[Self], config: DictConfig) -> Self:
        """Parse an omegaconf config into a CollateConfigs instance."""
        return cls(
            raffle_path=pathlib.Path(config.raffle_path),
            work_dir=pathlib.Path(config.work_dir),
            cache_dir=pathlib.Path(config.cache_dir),
        )


@dataclasses.dataclass
class TrainWithIndexUpdatesConfigs:
    """Models the configuration for a workflow that trains a model and periodically indexes the data."""

    dataset: DatasetsConfig
    tokenizer: TokenizerConfig
    dataloaders: DataLoaderConfigs
    collates: CollateConfigs
    trainer: TrainerConfig
    benchmark: BenchmarkConfig
    batch_size: BatchSizeConfig
    sys: SysConfig
    dl_sampler: Optional[SamplerFactoryConfig | list[SamplerFactoryConfig]] = None

    @classmethod
    def parse(cls: Type[Self], config: DictConfig) -> Self:
        """Parse an omegaconf config into a TrainWithIndexConfigs instance."""
        return cls(
            dataset=DatasetsConfig.parse(config.dataset),
            trainer=TrainerConfig.parse(config.trainer),
            benchmark=BenchmarkConfig.parse(config.benchmark),
            batch_size=BatchSizeConfig.parse(config.batch_size),
            dataloaders=DataLoaderConfigs.parse(config.dataloaders),
            collates=CollateConfigs.parse(config.collates),
            sys=SysConfig.parse(config.sys),
            dl_sampler=_parse_dl_sampler(config.dl_sampler),
            tokenizer=hydra.utils.instantiate(config.tokenizer),
        )


def _parse_dl_sampler(
    config: DictConfig | ListConfig | None,
) -> Optional[SamplerFactoryConfig | list[SamplerFactoryConfig]]:
    """Parse an omegaconf config into a SamplerFactoryConfig instance."""
    if config is None or len(config) == 0:
        return None

    if isinstance(config, (ListConfig, list, tuple, set)):
        samplers = [_parse_dl_sampler(c) for c in config]
        return [s for s in samplers if s is not None]  # type: ignore

    return SamplerFactoryConfig(**config)  # type: ignore
