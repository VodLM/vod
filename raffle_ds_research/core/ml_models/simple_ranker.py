from typing import Any, Optional

import datasets.fingerprint
import torch.nn
from transformers import BertModel

from raffle_ds_research.tools import pipes


class SimpleRanker(torch.nn.Module):
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

    def get_output_shape(self, *args, **kwargs) -> tuple[int, ...]:
        return (self.model.pooler.dense.out_features,)

    def forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        output = {}
        for field in self._fields:
            keys = [f"{field}.{key}" for key in self._keys]
            if all(key in batch for key in keys):
                args = [batch[key] for key in keys]
                model_output = self.model(*args)
                output_vector = model_output.pooler_output
                output[f"{field}.{self._output_vector_name}"] = output_vector

        return output


@datasets.fingerprint.hashregister(SimpleRanker)
def _hash_simple_ranker(hasher: datasets.fingerprint.Hasher, value: SimpleRanker):
    return pipes.fingerprint_torch_module(hasher, value)
