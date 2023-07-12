from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
import transformers

from raffle_ds_research.tools.pipes.utils.misc import pack_examples

_TOKENIZED_KEYS = ["input_ids", "attention_mask"]


def _get_token_keys(prefix_key: str) -> list[str]:
    return [f"{prefix_key}{k}" for k in _TOKENIZED_KEYS]


def _torch_pad_tokenized_field(
    batch: dict[str, Any],
    idx: Optional[list[int]] = None,  # noqa: ARG
    *,
    prefix_key: Optional[str] = None,
    tokenizer: transformers.PreTrainedTokenizerBase,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Pad a tokenized field."""
    prefix_key = prefix_key or ""
    tokenized_keys = _get_token_keys(prefix_key)
    if not all(k in batch for k in tokenized_keys):
        raise KeyError(f"Missing keys in batch: {tokenized_keys}. Found: {list(batch.keys())}")
    if all(isinstance(v, torch.Tensor) for v in batch.values()):
        return {k: batch[k] for k in tokenized_keys}

    # pad the tokenized field
    input_ids = batch[f"{prefix_key}input_ids"]
    attention_mask = batch[f"{prefix_key}attention_mask"]
    encodings = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        return_tensors="pt",
        **kwargs,
    )
    return dict(encodings)


def tokenize_pipe(
    batch: dict[str, Any],
    idx: Optional[list[int]] = None,  # noqa: ARG001
    *,
    text_key: str = "text",
    prefix_key: Optional[str] = None,
    tokenizer: transformers.PreTrainedTokenizerBase,
    **kwargs: Any,
) -> dict[str, Any]:
    """Tokenize a text field."""
    text = batch[text_key]
    encodings = tokenizer(text, **kwargs)
    prefix_key = prefix_key or ""
    return {f"{prefix_key}{k}": v for k, v in encodings.items()}


def torch_tokenize_pipe(
    batch: dict[str, Any],
    idx: Optional[list[int]] = None,
    *,
    text_key: str = "text",
    prefix_key: Optional[str] = None,
    tokenizer: transformers.PreTrainedTokenizerBase,
    lazy: bool = True,
    padding: bool | str = True,
    return_token_type_ids: bool = False,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Tokenize a text field."""
    prefix_key = prefix_key or ""
    tokenized_keys = ["input_ids", "attention_mask"]
    tokenized_keys = [f"{prefix_key}{k}" for k in tokenized_keys]
    if lazy and all(k in batch for k in tokenized_keys):
        return _torch_pad_tokenized_field(
            batch,
            idx,
            text_key=text_key,
            prefix_key=prefix_key,
            tokenizer=tokenizer,
            **kwargs,
        )

    text = batch[text_key]
    encodings = tokenizer(
        text,
        return_tensors="pt",
        padding=padding,
        return_token_type_ids=return_token_type_ids,
        **kwargs,
    )
    return {f"{prefix_key}{k}": v for k, v in encodings.items()}


def torch_tokenize_collate(
    examples: Iterable[dict[str, Any]],
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
    text_key: str = "text",
    prefix_key: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Collate a tokenized field."""
    prefix_key = prefix_key or ""
    tokenized_keys = _get_token_keys(prefix_key)
    first_eg, *other_egs = examples
    need_keys = tokenized_keys if set(tokenized_keys).issubset(first_eg) else [text_key]

    if not all(k in first_eg for k in need_keys):
        raise KeyError(f"Missing keys in batch: {need_keys}. Found: {list(first_eg.keys())}")

    batch = pack_examples(examples, keys=need_keys)
    return torch_tokenize_pipe(batch, text_key=text_key, prefix_key=prefix_key, tokenizer=tokenizer, **kwargs)
