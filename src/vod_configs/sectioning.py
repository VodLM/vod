import typing as typ

from .utils import StrictModel


class BaseSectioningConfig(StrictModel):
    """Base class for sectioning configurations."""

    mode: typ.Literal["sentence", "fixed_length"]
    section_template: str = r"{{ content }}"


class SentenceSectioningConfig(BaseSectioningConfig):
    """Sentence sectioning configuration."""

    mode: typ.Literal["sentence"] = "sentence"
    delimiter: str = ". "


class FixedLengthSectioningConfig(BaseSectioningConfig):
    """Fixed-length sectioning configuration."""

    mode: typ.Literal["fixed_length"] = "fixed_length"
    tokenizer_name_or_path: str
    max_length: int
    stride: int
    ellipsis_start: str = "(...) "
    ellipsis_end: str = " (...)"
    add_special_tokens: bool = True


SectioningConfig = SentenceSectioningConfig | FixedLengthSectioningConfig
