from __future__ import annotations

from typing import Any, Optional

import datasets.fingerprint
import torch.nn
from transformers import BertModel
from vod_tools import pipes


class SimpleRanker(torch.nn.Module):
    """A simple ranker that takes a question and a section and returns a vector. Useful for testing."""

    _fields: list[str] = ["question", "section"]
    _keys: list[str] = ["input_ids", "attention_mask"]
    _output_vector_name: str = "vector"

    def __init__(self, model: BertModel, fields: Optional[list[str]] = None, vector_name: Optional[str] = None):
        super().__init__()
        self.model = model
        if fields is not None:
            self._fields = fields
        if vector_name is not None:
            self._output_vector_name = vector_name

    def get_output_shape(self, *args: Any, **kwargs: Any) -> tuple[int, ...]:
        """Return the shape of the output vectors."""
        return (self.model.pooler.dense.out_features,)

    def forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Compute the embeddings for the input questions and sections."""
        output = {}
        for field in self._fields:
            keys = [f"{field}.{key}" for key in self._keys]
            if all(key in batch for key in keys):
                args = [batch[key] for key in keys]
                model_output = self.model(*args)
                output_vector = model_output.pooler_output
                output[f"{field}.{self._output_vector_name}"] = output_vector
        if not output:
            raise ValueError(f"Missing keys in batch: fields={self._fields}. Found: {list(batch.keys())}")

        return output


@datasets.fingerprint.hashregister(SimpleRanker)
def _hash_simple_ranker(hasher: datasets.fingerprint.Hasher, value: SimpleRanker) -> str:
    return pipes.fingerprint_torch_module(hasher, value)
