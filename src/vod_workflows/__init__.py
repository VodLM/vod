"""Collection of workflows such as training, evaluation, etc."""
from __future__ import annotations

from .evaluation.evaluation import benchmark
from .support.precompute import compute_vectors
from .training.train_with_updates import train_with_index_updates
