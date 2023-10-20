import re
import typing as typ

import pydantic
import transformers
from typing_extensions import Self, Type

from .utils.base import StrictModel


class DataLoaderConfig(StrictModel):
    """Base configuration for a pytorch DataLoader."""

    batch_size: int
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: None | int = None
    persistent_workers: bool = False


class TemplatesConfig(StrictModel):
    """Prompt templates."""

    query: str = pydantic.Field(
        default=r"query: {{ query }}",
        description="Template for formatting a query",
    )
    section: str = pydantic.Field(
        default=r"passage: {{ content }}",
        description="Template formatting documents before encoding for retrieval.",
    )
    lm: str = pydantic.Field(
        default=r"passage: {{ content }}\ninstructions: {{ query }}\nanswer: {{ answer }}",
        description="Template formatting inputs to the language model.",
    )

    @property
    @classmethod
    def input_variables(cls: Type[Self]) -> set[str]:
        """Return the input variables."""
        variables = set()
        for attribute_value in cls.__dict__.values():
            matches = re.findall(r"{{\s*(.*?)\s*}}", attribute_value)
            variables.update(matches)
        return variables


class TokenizerConfig(StrictModel):
    """Configuration for a tokenizer."""

    name_or_path: str
    use_fast: None | bool = True
    max_length: None | int = None
    truncation_side: None | typ.Literal["left", "right"] = None
    padding_side: None | typ.Literal["left", "right"] = None
    # keyword arguments for `tokenizer(..., **kwargs)`
    add_special_tokens: bool = True
    padding: typ.Literal["longest", "max_length", "do_not_pad"] = "longest"
    truncation: None | bool | typ.Literal[
        "longest_first", "only_first", "only_second", "do_not_truncate"
    ] = "longest_first"
    return_token_type_ids: bool = False

    def instantiate(
        self,
        **kws: dict[str, typ.Any],
    ) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
        """Instantiate the tokenizer."""
        if self.use_fast is not None:
            kws["use_fast"] = self.use_fast  # type: ignore
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.name_or_path, **kws)

        # Set configuration
        if self.max_length is not None:
            tokenizer.model_max_length = self.max_length
        if self.truncation_side is not None:
            tokenizer.truncation_side = self.truncation_side
        if self.padding_side is not None:
            tokenizer.padding_side = self.padding_side
        return tokenizer

    def kwargs(self) -> dict[str, typ.Any]:
        """Return the base kewword args."""
        return {
            "add_special_tokens": self.add_special_tokens,
            "padding": self.padding,
            "truncation": self.truncation,
            "return_token_type_ids": self.return_token_type_ids,
        }


class _BaseCollateConfig(StrictModel):
    """Defines a base configuration for the collate function."""

    templates: TemplatesConfig = pydantic.Field(default_factory=TemplatesConfig)


class TokenizerCollateConfig(_BaseCollateConfig):
    """Defines a base configuration for the collate function."""

    tokenizer: TokenizerConfig


class RealmCollateConfig(_BaseCollateConfig):
    """Defines a configuration for the retrieval collate function."""

    tokenizer_encoder: TokenizerConfig
    tokenizer_lm: None | TokenizerConfig
    # Realm specifics
    prefetch_n_sections: int = 100
    n_sections: None | int = 10
    max_pos_sections: None | int = 3
    support_size: None | int = None
    do_sample: bool = False
    in_batch_negatives: bool = False
    in_batch_neg_offset: int = 0
    prep_num_proc: int = 4
    lookup_engine: str = "sparse"  # Name of the search engine to use to lookup gold sections

    @pydantic.field_validator("tokenizer_lm", mode="before")
    @classmethod
    def _validate_tokenizer_config(
        cls: Type[Self], x: None | typ.Mapping[str, typ.Any]
    ) -> None | typ.Mapping[str, typ.Any]:
        if x is None:
            return None
        try:
            name_or_path = x.get("name_or_path", None)
            if name_or_path is None:
                return None
            return x
        except Exception:
            return x


class SamplerFactoryConfig(StrictModel):
    """Configuration for a dataloader sampler."""

    mode: typ.Literal["lookup", "inverse_frequency"]
    key: str
    lookup: None | dict[str, float] = None
    default_weight: float = 1.0
