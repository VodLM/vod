from __future__ import annotations

import json
from typing import Optional

from datasets.fingerprint import hashregister
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from .pipe import Pipe


class TokenizerPipe(Pipe):
    tokenizer: PreTrainedTokenizerBase = Field(..., alias="tokenizer")
    input_key: str = Field("text", alias="input_key")

    def _process_batch(self, batch: dict, idx: Optional[list[int]] = None, **kwargs) -> dict:
        """Process a batch."""
        text = batch[self.input_key]
        encodings = self.tokenizer(text, **kwargs)
        return {f"{self.input_key}.{k}": v for k, v in encodings.items()}


@hashregister(TokenizerPipe)
def _hash_tokenizer(hasher, value: TokenizerPipe):
    data = value.dict()

    def get_hps(tokenizer: PreTrainedTokenizerBase) -> dict:
        return {
            "obj": tokenizer.__class__.__name__,
            "name": tokenizer.name_or_path,
            "repr": tokenizer.__repr__(),
        }

    data["tokenizer"] = get_hps(data["tokenizer"])
    h = hasher.hash_bytes(json.dumps(data, sort_keys=True).encode("utf-8"))
    return h
