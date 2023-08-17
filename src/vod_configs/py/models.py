from __future__ import annotations

from typing import Any, Optional

import transformers

from .utils import StrictModel


class TokenizerConfig(StrictModel):
    """Configuration for a tokenizer."""

    model_name: str
    use_fast: Optional[bool] = True

    def instantiate(
        self,
        **kwargs: dict[str, Any],
    ) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
        """Instantiate the tokenizer."""
        if self.use_fast is not None:
            kwargs["use_fast"] = self.use_fast  # type: ignore

        return transformers.AutoTokenizer.from_pretrained(self.model_name, **kwargs)
