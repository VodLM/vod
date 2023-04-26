from __future__ import annotations

from typing import Optional

import faiss

from raffle_ds_research.tools.index_tools.vector_handler import VectorType, vector_handler


def build_index(
    vectors: VectorType,
    *,
    factory_string: str,
    add_batch_size: Optional[int] = None,
    faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
) -> faiss.Index:
    """Build an index from a factory string."""
    vectors = vector_handler(vectors)
    if len(vectors.shape) != 2:  # noqa: PLR2004
        raise ValueError(f"Only 2D tensors can be handled. Found shape `{vectors.shape}`")
    vector_size = vectors.shape[-1]
    index = faiss.index_factory(vector_size, factory_string, faiss_metric)
    if add_batch_size is None:
        add_batch_size = vectors.shape[0]
    for i, (_, batch) in enumerate(vectors.iter_batches(add_batch_size)):
        if i == 0:
            index.train(batch)
        index.add(batch)

    if index.ntotal != len(vectors) or index.d != vectors.shape[1]:
        raise ValueError(
            f"Index size doesn't match the size of the vectors."
            f"Found vectors: `{vectors.shape}`, index: `{index.ntotal, index.d}`"
        )

    return index
