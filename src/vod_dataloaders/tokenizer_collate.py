import typing as typ

import torch
import transformers
import vod_configs
import vod_types as vt
from typing_extensions import Self, Type
from vod_tools.misc.template import Template


def render_template_and_tokenize(
    inputs: typ.Iterable[dict[str, typ.Any]] | dict[str, list[typ.Any]],
    template: Template,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prefix_key: str = "",
    **tokenizer_kws: typ.Any,
) -> dict[str, torch.Tensor]:
    """Render a template, tokenize the resulting text and append the `prefix_key`."""
    texts = template(inputs)
    outputs = tokenizer(texts, **tokenizer_kws, return_tensors="pt")
    return {f"{prefix_key}{k}": v for k, v in outputs.items()}


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
        return render_template_and_tokenize(
            inputs=inputs,
            template=self.template,
            tokenizer=self.tokenizer,
            prefix_key=self.prefix_key,
            **self.tokenizer_kws,
        )

    @classmethod
    def instantiate(cls: Type[Self], config: vod_configs.TokenizerCollateConfig, *, field: str) -> Self:
        """Initialize the collate function for the `predict` function."""
        template = getattr(config.templates, field, None)
        if template is None:
            raise ValueError(f"Missing template for field `{field}`. Found: `{config.templates}`")

        # init the collate_fn
        return cls(
            template=template,
            prefix_key=f"{field}__",
            tokenizer=config.tokenizer.instantiate(),
            tokenizer_kws=config.tokenizer.kwargs(),
        )
