"""Collection of workflows such as training, processing, and benchmarking."""

__version__ = "0.1.0"

from .evaluation.retrieval import benchmark_retrieval
from .processing.vectors import compute_vectors
from .training.train import spawn_search_and_train
