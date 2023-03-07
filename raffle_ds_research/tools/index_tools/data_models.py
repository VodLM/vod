from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from pydantic import BaseModel


@dataclasses.dataclass
class FaissNumpyResults:
    scores: np.ndarray
    indices: np.ndarray


class FaissSearchResults(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

    scores: list[list[float]]
    indices: list[list[int]]


class FaissInitConfig(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

    index_path: str
    nprobe: int = 8


class InitResponse(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

    success: bool
    exception: Optional[str]


class SearchFaissQuery(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

    vectors: list  # todo [list[float]]
    top_k: int = 3
