"""Wrappers for Raffle datasets."""
__version__ = "0.1.0"

from .interface import (
    load_dataset,
    load_queries,
    load_sections,
)
from .loaders import (
    BeirDatasetLoader,
)
