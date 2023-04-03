from __future__ import annotations

from typing import Optional, Union

import omegaconf
import pydantic

from raffle_ds_research.core.workflows.schedule import ScheduleConfig


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

    def get_weight(self, step: int) -> float:
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
