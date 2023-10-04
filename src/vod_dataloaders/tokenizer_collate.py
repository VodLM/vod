import typing as typ

import torch
import transformers
import vod_configs
import vod_types as vt
from typing_extensions import Self, Type
from vod_tools.misc.template import Template


class TokenizerCollate(vt.Collate[typ.Any, torch.Tensor]):
    """Collate function to format text and tokenize into `field.input_ids and `field.attention_mask` tensors."""

    def __init__(
        self,
        template: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
        prefix_key: str = "",
        tokenizer_kws: None | dict[str, typ.Any] = None,
    ) -> None:
        """Initialize the collate function."""
        self.template = Template(template)
        self.tokenizer = tokenizer
        self.prefix_key = prefix_key
        self.tokenizer_kws = tokenizer_kws or {}

    def __call__(
        self,
        inputs: typ.Iterable[dict[str, typ.Any]] | dict[str, list[typ.Any]],
        **kws: typ.Any,
    ) -> dict[str, torch.Tensor]:
        """Render a template, tokenize the resulting text and append the `prefix_key`."""
        texts = self.template(inputs)
        outputs = self.tokenizer(texts, **self.tokenizer_kws, return_tensors="pt")  # <- hard code return_tensors="pt"
        return {f"{self.prefix_key}{k}": v for k, v in outputs.items()}

    @classmethod
    def from_config(
        cls: Type[Self],
        config: vod_configs.BaseCollateConfig,
        *,
        field: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> Self:
        """Initialize the collate function for the `predict` function."""
        max_length = {
            "query": config.query_max_length,
            "section": config.section_max_length,
        }[field]
        template = getattr(config.templates, field, None)
        if template is None:
            raise ValueError(f"Missing template for field `{field}`. Found: `{config.templates}`")

        # init the collate_fn
        return cls(
            template=template,
            prefix_key=f"{field}.",
            tokenizer=tokenizer,
            tokenizer_kws={
                "max_length": max_length,
                "truncation": True,
                "padding": "max_length",
            },
        )
