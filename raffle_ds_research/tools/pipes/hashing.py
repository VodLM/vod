import functools
import json

import transformers
from datasets import fingerprint

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


def _register_special_hashers():
    """Register special hashers for some classes."""
    # transformers.PreTrainedTokenizer
    rules = [(_hash_tokenizer, _TOKENIZERS_CLASSES), (_hash_partial, [functools.partial])]

    for rule in rules:
        fingerprint.hashregister(*rule[1])(rule[0])


def _hash_tokenizer(hasher: fingerprint.Hasher, value: transformers.PreTrainedTokenizer):
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
    hashed_data = {k: hasher.hash(v) for k, v in data.items()}

    json_data = json.dumps(hashed_data, sort_keys=True)
    tokenizer_hash = hasher.hash(json_data)
    return tokenizer_hash


def _hash_partial(hasher: fingerprint.Hasher, value: functools.partial):
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
