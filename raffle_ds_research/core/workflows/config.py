from __future__ import annotations

import dataclasses
from typing import Optional, Union

import omegaconf
import pydantic
from omegaconf import DictConfig

from raffle_ds_research.core.workflows.schedule import ScheduleConfig
from raffle_ds_research.tools.utils import loader_config


class DefaultCollateConfig(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    n_sections: int
    max_pos_sections: int
    prefetch_n_sections: int
    do_sample: bool
    question_max_length: int = 512
    section_max_length: int = 512


class SearchConfig(pydantic.BaseModel):
    """Base configuration for search engines (e.g., bm25, faiss)."""

    class Config:
        """Pydantic configuration."""

        extra = pydantic.Extra.forbid

    schedule: Union[float, ScheduleConfig] = 1.0
    label_key: str = "kb_id"

    @pydantic.validator("schedule")
    def _validate_schedule(cls, v: dict | omegaconf.DictConfig | ScheduleConfig | float) -> float | ScheduleConfig:
        if isinstance(v, ScheduleConfig):
            return v
        if isinstance(v, (bool, float, int)):
            return float(v)
        elif isinstance(v, dict):
            return ScheduleConfig(**v)
        elif isinstance(v, omegaconf.DictConfig):
            return ScheduleConfig(**v)

        raise ValueError(f"Invalid schedule: {v}")

    def get_weight(self, step: float) -> float:
        """Get the weight for the given step."""
        if isinstance(self.schedule, float):
            return self.schedule

        if isinstance(self.schedule, ScheduleConfig):
            return self.schedule(step)

        raise TypeError(f"Invalid schedule: {self.schedule}")


class DefaultFaissConfig(SearchConfig):
    """Configures a faiss search engine."""

    factory: str
    nprobe: int


class DefaultBm25Config(SearchConfig):
    """Configures a bm25 search engine."""

    indexed_key: str = "text"
    use_labels: bool = True


class MultiIndexConfig(pydantic.BaseModel):
    """Configures a group of indexes (e.g., bm25, faiss)."""

    update_freq: Optional[Union[int, list[int]]]
    reset_model: bool = False
    faiss: DefaultFaissConfig
    bm25: DefaultBm25Config

    @pydantic.validator("update_freq", pre=True)
    def _validate_update_freq(cls, v: Union[int, list[int], omegaconf.ListConfig]) -> int | list[int]:
        if isinstance(v, omegaconf.ListConfig):
            v = omegaconf.OmegaConf.to_container(v)
        return v

    @pydantic.validator("faiss")
    def _validate_faiss(cls, v: dict | omegaconf.DictConfig | DefaultFaissConfig) -> DefaultFaissConfig:
        if isinstance(v, DefaultFaissConfig):
            return v
        if isinstance(v, (dict, omegaconf.DictConfig)):
            return DefaultFaissConfig(**v)

        raise ValueError(f"Invalid faiss config: {v}")

    @pydantic.validator("bm25")
    def _validate_bm25(cls, v: dict | omegaconf.DictConfig | DefaultBm25Config) -> DefaultBm25Config:
        if isinstance(v, DefaultBm25Config):
            return v
        if isinstance(v, (dict, omegaconf.DictConfig)):
            return DefaultBm25Config(**v)

        raise ValueError(f"Invalid bm25 config: {v}")


@dataclasses.dataclass
class DataLoaderConfigs:
    train: Optional[loader_config.DataLoaderConfig]
    eval: Optional[loader_config.DataLoaderConfig]
    predict: Optional[loader_config.DataLoaderConfig]

    @classmethod
    def parse(cls, config: DictConfig) -> "DataLoaderConfigs":
        return cls(
            train=loader_config.DataLoaderConfig(**config.train) if config.train is not None else None,
            eval=loader_config.DataLoaderConfig(**config.eval) if config.eval is not None else None,
            predict=loader_config.DataLoaderConfig(**config.predict) if config.predict is not None else None,
        )


@dataclasses.dataclass
class CollateConfigs:
    train: Optional[DefaultCollateConfig]
    eval: Optional[DefaultCollateConfig]
    static: Optional[DefaultCollateConfig]

    @classmethod
    def parse(cls, config: DictConfig) -> "CollateConfigs":
        return cls(
            train=DefaultCollateConfig(**config.train) if config.train is not None else None,
            eval=DefaultCollateConfig(**config.eval) if config.eval is not None else None,
            static=DefaultCollateConfig(**config.static) if config.static is not None else None,
        )


@dataclasses.dataclass
class TrainWithIndexConfigs:
    dataloaders: DataLoaderConfigs
    collates: CollateConfigs
    indexes: MultiIndexConfig

    @classmethod
    def parse(cls, config: DictConfig) -> "TrainWithIndexConfigs":
        dataloaders = DataLoaderConfigs.parse(config.dataloaders)
        collates = CollateConfigs.parse(config.collates)
        indexes_config = MultiIndexConfig(**config.indexes)

        return cls(dataloaders=dataloaders, collates=collates, indexes=indexes_config)
