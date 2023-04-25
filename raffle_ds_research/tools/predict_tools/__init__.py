"""A collection of utilities to store the predictions of a models (output vectors) to a file (`tensorstore`)."""
from .interface import predict  # noqa: F401
from .ts_utils import TensorStoreFactory
