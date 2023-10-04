import functools
import json
import typing as typ

import datasets
import torch
import transformers

from src.vod_tools.misc.tensor_tools import serialize_tensor

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


def register_special_hashers() -> None:
    """Register special hashers for some classes."""
    rules: list[tuple[typ.Callable, list[type]]] = [
        (_hash_tokenizer, _TOKENIZERS_CLASSES),
        (_hash_partial, [functools.partial]),
        (_hash_dataset, [datasets.Dataset]),
        (_hash_dataset_dict, [datasets.DatasetDict]),
        (_fingerprint_torch_module, [torch.nn.Module]),
    ]
    for rule in rules:
        datasets.fingerprint.hashregister(*rule[1])(rule[0])


def _hash_tokenizer(hasher: datasets.fingerprint.Hasher, value: transformers.PreTrainedTokenizer) -> str:
    """Implement a hash function for a pretrained tokenizer.

    The default hash of `transformers.PreTrainedTokenizer` is non-deterministic: it changes after a first iteration.
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
    return hasher.hash(json_data)


def _hash_partial(hasher: datasets.fingerprint.Hasher, value: functools.partial) -> str:
    """The default hash of `functools.partial`."""
    data = {
        "cls": value.__class__,
        "func": value.func,
        "args": value.args,
        "keywords": value.keywords,
    }

    hashed = {k: hasher.hash(v) for k, v in data.items() if k not in {"keywords"}}
    hashed["keywords"] = {k: hasher.hash(v) for k, v in data["keywords"].items()}  # type: ignore
    return hasher.hash(hashed)


def _fingerprint_torch_module(hasher: datasets.fingerprint.Hasher, value: torch.nn.Module) -> str:
    """Fingerprint a `torch.nn.Module`."""
    for k, v in value.state_dict().items():
        hasher.update(k)
        u = serialize_tensor(v)
        hasher.update(u)
    return hasher.hexdigest()


def fingerprint_torch_module(value: torch.nn.Module) -> str:
    """Fingerprint a `torch.nn.Module`."""
    hasher = datasets.fingerprint.Hasher()
    return _fingerprint_torch_module(hasher, value)


def _hash_dataset(_: datasets.fingerprint.Hasher, value: datasets.Dataset) -> str:
    return value._fingerprint


def _hash_dataset_dict(hasher: datasets.fingerprint.Hasher, value: datasets.DatasetDict) -> str:
    values = {key: value[key]._fingerprint for key in value}
    return hasher.hash(json.dumps(values))
