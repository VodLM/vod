from __future__ import annotations

import math
from typing import Optional

import datasets
import numpy as np
import torch
from datasets import fingerprint

from src.vod_tools import dstruct, interfaces, pipes
from src.vod_tools.misc.tensor_tools import serialize_tensor


def make_predict_fingerprint(
    *,
    dataset: dstruct.SizedDataset | datasets.Dataset,
    collate_fn: pipes.Collate,
    model: torch.nn.Module | interfaces.ProtocolEncoder,
    model_output_key: Optional[str] = None,
) -> str:
    """Make a fingerprint for the `predict` operation."""
    dset_fingerprint = _get_dset_fingerprint(dataset)
    model_fingerprint = _get_model_fingerprint(model)
    collate_fn_fingerprint = _get_collate_fn_fingerprint(collate_fn)
    op_fingerprint = f"{dset_fingerprint}_{model_fingerprint}_{collate_fn_fingerprint}"
    if model_output_key:
        op_fingerprint += f"_{model_output_key}"
    return op_fingerprint


def _get_model_fingerprint(model: torch.nn.Module | interfaces.ProtocolEncoder) -> str:
    if isinstance(model, torch.nn.Module):
        state = model.state_dict()
        hasher = fingerprint.Hasher()
        hasher.update(type(model).__name__)
        for k, v in sorted(state.items(), key=lambda x: x[0]):
            hasher.update(k)
            u = serialize_tensor(v)
            hasher.update(u)
        return hasher.hexdigest()

    return fingerprint.Hasher.hash(model)


def _get_collate_fn_fingerprint(collate_fn: pipes.Collate) -> str:
    return fingerprint.Hasher.hash(collate_fn)


def _get_dset_fingerprint(dataset: dstruct.SizedDataset | datasets.Dataset, max_samples: float = math.inf) -> str:
    if isinstance(dataset, datasets.Dataset):
        return dataset._fingerprint

    hasher = fingerprint.Hasher()
    hasher.update({"length": len(dataset)})

    # select random row ids
    if len(dataset) > max_samples:
        rgn = np.random.RandomState(0)
        ids = rgn.choice(len(dataset), size=int(max_samples), replace=False)
    else:
        ids = range(len(dataset))

    # hash the rows
    for i in ids:
        hasher.update(dataset[i])

    return hasher.hexdigest()
