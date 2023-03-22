from __future__ import annotations

import pydantic


class DefaultCollateConfig(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    n_sections: int
    max_pos_sections: int
    prefetch_n_sections: int
    sample_negatives: bool
    question_max_length: int = 512
    section_max_length: int = 512


class DefaultFaissConfig(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    factory: str
    nprobe: int
    enabled: bool = True


class DefaultBm25Config(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    enabled: bool = True
    indexed_key: str = "section"
