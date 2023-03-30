from __future__ import annotations

from typing import Union

import omegaconf
import pydantic

from raffle_ds_research.cli.utils.schedule import ScheduleConfig


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
    class Config:
        extra = pydantic.Extra.forbid

    schedule: Union[float, ScheduleConfig] = 1.0
    label_key: str = "kb_id"

    @pydantic.validator("schedule")
    def _validate_schedule(cls, v):
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
        if isinstance(self.schedule, float):
            return self.schedule
        elif isinstance(self.schedule, ScheduleConfig):
            return self.schedule(step)
        else:
            raise TypeError(f"Invalid schedule: {self.schedule}")


class DefaultFaissConfig(SearchConfig):
    factory: str
    nprobe: int


class DefaultBm25Config(SearchConfig):
    indexed_key: str = "text"
    use_labels: bool = True
