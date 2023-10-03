import copy
import typing as typ

import peft
import transformers
from typing_extensions import Self

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


class ModelOptimConfig(StrictModel):
    """Configure the model optimizations."""

    compile: bool = False
    compile_kwargs: dict[str, typ.Any] = {}
    gradient_checkpointing: bool = False
    prepare_for_kbit_training: bool = False
    peft_config: None | peft.config.PeftConfig = None
    params_dtype: None | str = None

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
