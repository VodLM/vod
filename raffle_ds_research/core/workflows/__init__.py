"""Collection of workflows such as training, evaluation, etc."""
from __future__ import annotations

from .evaluation import benchmark
from .precompute import compute_vectors
from .train_with_updates import train_with_index_updates
