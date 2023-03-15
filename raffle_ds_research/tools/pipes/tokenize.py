from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
import transformers

from raffle_ds_research.tools.pipes.utils.misc import pack_examples

_TOKENIZED_KEYS = ["input_ids", "attention_mask"]


def _get_token_keys(field):
    tokenized_keys = [f"{field}.{k}" for k in _TOKENIZED_KEYS]
    return tokenized_keys


def _torch_pad_tokenized_field(
    batch,
    idx: Optional[list[int]] = None,
    *,
    field: str,
    tokenizer: transformers.PreTrainedTokenizer,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Pad a tokenized field."""
    tokenized_keys = _get_token_keys(field)
    if not all(k in batch for k in tokenized_keys):
        raise KeyError(f"Missing keys in batch: {tokenized_keys}. Found: {list(batch.keys())}")
    if all(isinstance(v, torch.Tensor) for v in batch.values()):
        return {k: batch[k] for k in tokenized_keys}

    # pad the tokenized field
    input_ids = batch[f"{field}.input_ids"]
    attention_mask = batch[f"{field}.attention_mask"]
    encodings = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        return_tensors="pt",
        **kwargs,
    )
    return dict(encodings)


def tokenize_pipe(
    batch: dict[str, Any],
    idx: Optional[list[int]] = None,
    *,
    field: str,
    tokenizer: transformers.PreTrainedTokenizer,
    **kwargs: Any,
) -> dict[str, Any]:
    """Tokenize a text field."""
    text = batch[field]
    encodings = tokenizer(text, **kwargs)
    return {f"{field}.{k}": v for k, v in encodings.items()}


def torch_tokenize_pipe(
    batch: dict[str, Any],
    idx: Optional[list[int]] = None,
    *,
    field: str,
    tokenizer: transformers.PreTrainedTokenizer,
    lazy: bool = True,
    padding: bool = True,
    return_token_type_ids: bool = False,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Tokenize a text field."""
    tokenized_keys = ["input_ids", "attention_mask"]
    tokenized_keys = [f"{field}.{k}" for k in tokenized_keys]
    if lazy and all(k in batch for k in tokenized_keys):
        return _torch_pad_tokenized_field(
            batch,
            idx,
            field=field,
            tokenizer=tokenizer,
            **kwargs,
        )

    text = batch[field]
    encodings = tokenizer(
        text,
        return_tensors="pt",
        padding=padding,
        return_token_type_ids=return_token_type_ids,
        **kwargs,
    )
    return {f"{field}.{k}": v for k, v in encodings.items()}


def torch_tokenize_collate(
    examples: Iterable[dict[str, Any]],
    *,
    field: str,
    tokenizer: transformers.PreTrainedTokenizer,
    **kwargs: Any,
) -> dict[str, Any]:
    """Collate a tokenized field."""
    tokenized_keys = _get_token_keys(field)
    first_eg, *other_egs = examples
    if set(tokenized_keys).issubset(first_eg):
        need_keys = tokenized_keys
    else:
        need_keys = [field]

    if not all(k in first_eg for k in need_keys):
        raise KeyError(f"Missing keys in batch: {need_keys}. Found: {list(first_eg.keys())}")

    batch = pack_examples(examples, keys=need_keys)
    return torch_tokenize_pipe(batch, field=field, tokenizer=tokenizer, **kwargs)
