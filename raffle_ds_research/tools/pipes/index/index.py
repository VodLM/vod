from __future__ import annotations

import math
from typing import Optional, TypeVar

from datasets import Dataset as HfDataset
from pydantic import BaseModel, Field, PrivateAttr, root_validator

from ..pipe import E, I, O, Pipe
from ..utils.misc import keep_only_columns

T = TypeVar("T")


class Index(Pipe[I, O, E]):
    """An index holds a _section dataset"""

    _required_sections_columns: Optional[set[str]] = PrivateAttr(None)
    _sections: HfDataset = PrivateAttr(None)
    pid_key: str = Field("section.pid", description="The output key for section indices")
    score_key: str = Field("section.score", description="The output key for section scores")
    label_key: str = Field("section.label", description="The output key for section labels")
    max_top_k: Optional[int] = Field(None, description="The number of top k to return")
    padding: bool = Field(False, description="Whether to pad to `max_top_k` by sampling random elements")
    padding_score: float = Field(-math.inf, description="The score to use for padding")

    @root_validator
    def _check_padding(cls, values):
        if values["padding"] and values["max_top_k"] is None:
            raise ValueError("`max_top_k` must be set if `padding` is set")
        return values

    def __init__(self, sections: HfDataset, **kwargs):
        super().__init__(**kwargs)
        if self._required_sections_columns is not None:
            sections = keep_only_columns(sections, self._required_sections_columns, strict=True)
        self._sections = sections
        self._build_index()

    def _build_index(self):
        """Build the index (e.g., faiss)"""
        ...


class SupervisedIndexInput(BaseModel):
    section_id: list[Optional[int]] = Field(..., description="The section ids")
    answer_id: list[int] = Field(..., description="The answer ids")


class SupervisedIndexOuput(BaseModel):
    section_pid: list = Field(
        ...,
        description="The ids of the sections within the `sections` dataset. " "Shape [bs, n_sections]",
        alias="section.pid",
    )
    section_score: list = Field(
        ...,
        description="The scores  of the sections within the `sections` dataset. " "Shape [bs, n_sections]",
        alias="section.score",
    )
    section_label: list = Field(
        ...,
        description="Whether a section is a positive example. " "Shape [bs, n_sections]",
        alias="section.label",
    )
