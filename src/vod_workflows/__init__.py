"""Collection of workflows such as training, processing, and benchmarking."""


from .evaluation.retrieval import benchmark_retrieval
from .processing.vectors import compute_vectors
from .training.training import spawn_search_and_train
