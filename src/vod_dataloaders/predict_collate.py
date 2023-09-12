from __future__ import annotations

from typing import Any, Iterable

import transformers
import vod_configs
from typing_extensions import Self, Type
from vod_tools import pipes
from vod_tools.misc.template import Template


class PredictCollate(pipes.Collate):
    """Collate function for the `predict` step."""

    def __init__(
        self,
        template: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
        prefix_key: str = "",
        tokenizer_kws: None | dict[str, Any] = None,
    ) -> None:
        """Initialize the collate function."""
        self.template = Template(template)
        self.tokenizer = tokenizer
        self.prefix_key = prefix_key
        self.tokenizer_kws = tokenizer_kws or {}

    def __call__(
        self,
        examples: Iterable[dict[str, Any]] | dict[str, list[Any]],
        **kws: Any,
    ) -> dict[str, Any]:
        """Render a template and tokenize the result."""
        if isinstance(examples, dict):
            texts = self.template.render_batch(examples)  # type: ignore
        else:
            texts = [self.template.render(ex) for ex in examples]
        # tokenize the result
        outputs = self.tokenizer(texts, **self.tokenizer_kws)
        # prefix the keys
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
                "return_tensors": "pt",
            },
        )
