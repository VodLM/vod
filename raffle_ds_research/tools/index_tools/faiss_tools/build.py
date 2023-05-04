from __future__ import annotations

from typing import Optional

import faiss
import numpy as np

from raffle_ds_research.tools import dstruct


def build_faiss_master(
    vectors: dstruct.SizedDataset[np.ndarray],
    *,
    factory_string: str,
    train_size: Optional[int] = None,
    faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
) -> faiss.Index:
    """Build an index from a factory string."""
    vector_shape = vectors[0].shape
    if len(vector_shape) > 1:  # noqa: PLR2004
        raise ValueError(f"Only 1D vectors can be handled. Found shape `{vector_shape}`")
    vector_size = vector_shape[-1]
    index = faiss.index_factory(vector_size, factory_string, faiss_metric)

    if train_size is None:
        train_size = len(vectors)

    for i in range(0, len(vectors), train_size):
        batch = vectors[i : i + train_size]
        if i == 0:
            index.train(batch)
        index.add(batch)

    if index.ntotal != len(vectors) or index.d != vector_size:
        raise ValueError(
            f"Index size doesn't match the size of the vectors."
            f"Found vectors: `{vector_size}`, index: `{index.ntotal, index.d}`"
        )

    return index
