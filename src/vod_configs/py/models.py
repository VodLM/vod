import typing as typ

import transformers

from .utils import StrictModel


class TokenizerConfig(StrictModel):
    """Configuration for a tokenizer."""

    name_or_path: str
    use_fast: None | bool = True

    def instantiate(
        self,
        **kws: dict[str, typ.Any],
    ) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
        """Instantiate the tokenizer."""
        if self.use_fast is not None:
            kws["use_fast"] = self.use_fast  # type: ignore

        return transformers.AutoTokenizer.from_pretrained(self.name_or_path, **kws)
