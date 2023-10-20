"""Collection of workflows such as training, processing, and benchmarking."""

__version__ = "0.2.0"

from .utils.schemas import (
    QueriesWithVectors,
    SectionsWithVectors,
)
from .workflows.benchmark import benchmark_retrieval
from .workflows.compute import compute_vectors
from .workflows.train import spawn_search_and_train
