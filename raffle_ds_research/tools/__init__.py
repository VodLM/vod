"""A collection of Python tools with minimum interdependencies."""
from __future__ import annotations

from .predict_tools.interface import predict  # noqa: F401
from .predict_tools.ts_utils import TensorStoreFactory  # noqa: F401
