"""A collection of utilities to store the predictions of a models (output vectors) to a file (`tensorstore`)."""
from __future__ import annotations

from .interface import predict  # noqa: F401
from .ts_utils import TensorStoreFactory
