import functools
import json
from typing import Callable

import torch
import transformers
from datasets import fingerprint

from raffle_ds_research.tools.utils.tensor_tools import serialize_tensor

_TOKENIZERS_CLASSES = [
    transformers.PreTrainedTokenizer,
    transformers.PreTrainedTokenizerFast,
    transformers.BertTokenizer,
    transformers.BertTokenizerFast,
    transformers.T5Tokenizer,
    transformers.T5TokenizerFast,
    transformers.GPT2Tokenizer,
    transformers.GPT2TokenizerFast,
]


def _register_special_hashers() -> None:
    """Register special hashers for some classes."""
    rules: list[tuple[Callable, list[type]]] = [
        (_hash_tokenizer, _TOKENIZERS_CLASSES),
        (_hash_partial, [functools.partial]),
    ]
    for rule in rules:
        fingerprint.hashregister(*rule[1])(rule[0])


def _hash_tokenizer(hasher: fingerprint.Hasher, value: transformers.PreTrainedTokenizer) -> str:
    """The default hash of `transformers.PreTrainedTokenizer` is non-deterministic: it changes after a first iteration.
    Implement a custom hash function that is deterministic.
    """
    data = {
        "cls": value.__class__,
        "vocab": value.get_vocab(),
        "model_max_length": value.model_max_length,
        "padding_side": value.padding_side,
        "tuncation_side": value.truncation_side,
    }
    data["vocab"] = json.dumps(data["vocab"], sort_keys=True)
    hashed_data = {k: hasher.hash(v) for k, v in data.items()}
    json_data = json.dumps(hashed_data, sort_keys=True)
    tokenizer_hash = hasher.hash(json_data)
    return tokenizer_hash


def _hash_partial(hasher: fingerprint.Hasher, value: functools.partial) -> str:
    """The default hash of `functools.partial`."""
    data = {
        "cls": value.__class__,
        "func": value.func,
        "args": value.args,
        "keywords": value.keywords,
    }

    hashed = {k: hasher.hash(v) for k, v in data.items() if k not in {"keywords"}}
    hashed["keywords"] = {k: hasher.hash(v) for k, v in data["keywords"].items()}
    return hasher.hash(hashed)


def fingerprint_torch_module(hasher: fingerprint.Hasher, value: torch.nn.Module) -> str:
    hasher = fingerprint.Hasher()
    for k, v in value.state_dict().items():
        hasher.update(k)
        u = serialize_tensor(v)
        hasher.update(u)
    return hasher.hexdigest()
