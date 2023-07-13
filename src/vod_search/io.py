from __future__ import annotations

import base64
import io
from typing import Optional, Type

import numpy as np
import torch


def _bytes_to_unicode(bytes_obj: bytes) -> str:
    return base64.urlsafe_b64encode(bytes_obj).decode("utf-8")


def _unicode_to_bytes(unicode_obj: str) -> bytes:
    return base64.urlsafe_b64decode(unicode_obj)


def serialize_np_array(array: np.ndarray) -> str:
    """Serializes a numpy array to a string."""
    bytes_buffer = io.BytesIO()
    np.save(bytes_buffer, array, allow_pickle=True)  # type: ignore
    bytes_buffer = bytes_buffer.getvalue()
    return _bytes_to_unicode(bytes_buffer)


def deserialize_np_array(encoded_array: str, *, dtype: Optional[Type[np.dtype]] = None) -> np.ndarray:
    """Deserializes a numpy array from a string."""
    np_bytes = _unicode_to_bytes(encoded_array)
    load_bytes = io.BytesIO(np_bytes)
    loaded_np = np.load(load_bytes, allow_pickle=True)  # type: ignore
    if dtype is not None:
        loaded_np = loaded_np.astype(dtype)
    return loaded_np


def serialize_torch_tensor(tensor: torch.Tensor) -> str:
    """Serializes a torch tensor to a string."""
    bytes_buffer = io.BytesIO()
    torch.save(tensor, bytes_buffer)
    bytes_buffer = bytes_buffer.getvalue()
    return base64.urlsafe_b64encode(bytes_buffer).decode("utf-8")


def deserialize_torch_tensor(encoded_tensor: str, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Deserializes a torch tensor from a string."""
    bytes_buffer = base64.urlsafe_b64decode(encoded_tensor)
    bytes_buffer = io.BytesIO(bytes_buffer)
    deserialized_ts = torch.load(bytes_buffer)
    if dtype is not None:
        deserialized_ts = deserialized_ts.to(dtype)
    return deserialized_ts
