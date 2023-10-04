import math
import typing as typ

import datasets
import numpy as np
import torch
import vod_types as vt
from datasets import fingerprint
from vod_tools.misc.tensor_tools import serialize_tensor


def make_predict_fingerprint(
    *,
    dataset: typ.Any | vt.Sequence | vt.SupportsGetFingerprint,
    collate_fn: typ.Any | vt.Collate | vt.SupportsGetFingerprint,
    model: typ.Any | torch.nn.Module | vt.SupportsGetFingerprint,
    model_output_key: None | str = None,
) -> str:
    """Make a fingerprint for the `predict` operation."""
    dset_fingerprint = _get_dset_fingerprint(dataset)
    model_fingerprint = _get_model_fingerprint(model)
    collate_fn_fingerprint = _get_collate_fn_fingerprint(collate_fn)
    op_fingerprint = f"{dset_fingerprint}_{model_fingerprint}_{collate_fn_fingerprint}"
    if model_output_key:
        op_fingerprint += f"_{model_output_key}"
    return op_fingerprint


def use_get_fingerprint_if_available(
    fun: typ.Callable[[typ.Any], str]
) -> typ.Callable[[typ.Any | vt.SupportsGetFingerprint], str]:
    """Decorate a `get_fingerprint` function to first try call `x.get_fingerprint()`."""

    def wrapper(x: typ.Any) -> str:  # noqa: ANN401
        try:
            return x.get_fingerprint()
        except AttributeError:
            return fun(x)

    return wrapper


@use_get_fingerprint_if_available
def _get_model_fingerprint(model: torch.nn.Module) -> str:
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


@use_get_fingerprint_if_available
def _get_collate_fn_fingerprint(collate_fn: typ.Any) -> str:  # noqa: ANN401
    return fingerprint.Hasher.hash(collate_fn)


@use_get_fingerprint_if_available
def _get_dset_fingerprint(
    dataset: vt.Sequence | datasets.Dataset,
    max_samples: float = math.inf,
) -> str:
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
