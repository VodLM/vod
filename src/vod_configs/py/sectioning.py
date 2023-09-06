from __future__ import annotations

from typing import Literal, Union

from .utils import StrictModel


class BaseSectioningConfig(StrictModel):
    """Base class for sectioning configurations."""

    mode: Literal["sentence", "fixed_length"]
    section_template: str = r"{{ content }}"


class SentenceSectioningConfig(BaseSectioningConfig):
    """Sentence sectioning configuration."""

    mode: Literal["sentence"]
    delimiter: str = ". "


class FixedLengthSectioningConfig(BaseSectioningConfig):
    """Fixed-length sectioning configuration."""

    mode: Literal["fixed_length"]
    tokenizer_name_or_path: str
    max_length: int
    stride: int
    ellipsis_start: str = "(...) "
    ellipsis_end: str = " (...)"
    add_special_tokens: bool = True


SectioningConfig = Union[SentenceSectioningConfig, FixedLengthSectioningConfig]
