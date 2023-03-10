from __future__ import annotations

import faiss

# faiss.contrib.torch_utils: required to handle torch.Tensor inputs.
from faiss.contrib import torch_utils  # type: ignore

from raffle_ds_research.tools.index_tools.vector_handler import VectorType, vector_handler


def build_index(
    vectors: VectorType, factory_string: str, add_batch_size: int = 1000, faiss_metric: int = faiss.METRIC_INNER_PRODUCT
) -> faiss.Index:
    """Build an index from a factory string"""
    vectors = vector_handler(vectors)
    if len(vectors.shape) != 2:
        raise ValueError(f"Only 2D tensors can be handled. Found shape `{vectors.shape}`")
    vector_size = vectors.shape[-1]
    index = faiss.index_factory(vector_size, factory_string, faiss_metric)
    for ids, batch in vectors.iter_batches(add_batch_size):
        index.add(batch)

    if index.ntotal != len(vectors) or index.d != vectors.shape[1]:
        raise ValueError(
            f"Index size doesn't match the size of the vectors."
            f"Found vectors: `{vectors.shape}`, index: `{index.ntotal, index.d}`"
        )

    return index
