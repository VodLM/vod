import copy
import typing as typ

import peft
from typing_extensions import Self

from .utils.base import StrictModel


class TweaksConfig(StrictModel):
    """Configure the model optimizations."""

    compile: bool = False
    compile_kwargs: dict[str, typ.Any] = {}
    gradient_checkpointing: bool = False
    prepare_for_kbit_training: bool = False
    peft_config: None | peft.config.PeftConfig = None
    force_dtype: None | str = None

    @classmethod
    def parse(cls: typ.Type[Self], **config: typ.Any) -> Self:
        """Parse a serialized configuration."""
        config = copy.deepcopy(config)
        if compile_kwargs := config.get("compile_kwargs", {}):
            config["compile_kwargs"] = {k: v for k, v in compile_kwargs.items() if v is not None}
        if peft_config := config.get("peft_config", None):
            if not isinstance(peft_config, peft.config.PeftConfig):
                if target_modules := peft_config.get("target_modules", None):
                    peft_config["target_modules"] = [str(x) for x in target_modules]
                peft_config = peft.mapping.get_peft_config(dict(**peft_config))
            config["peft_config"] = peft_config
        return cls(**config)


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
