from __future__ import annotations

from typing import Optional

import faiss
import numpy as np
import torch
from loguru import logger

from raffle_ds_research.tools import dstruct
from raffle_ds_research.tools.index_tools.faiss_tools import build_gpu, support


def build_faiss_index(
    vectors: dstruct.SizedDataset[np.ndarray],
    *,
    factory_string: str,
    train_size: Optional[int] = None,
    faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
    gpu_config: Optional[build_gpu.FaissGpuConfig] = None,
) -> faiss.Index:
    """Build an index from a factory string."""
    vector_shape = vectors[0].shape
    if len(vector_shape) > 1:  # noqa: PLR2004
        raise ValueError(f"Only 1D vectors can be handled. Found shape `{vector_shape}`")

    # Infer the number of centroids if needed.
    nvecs = len(vectors) if train_size is None else min(train_size, len(vectors))
    factory_string = support.infer_factory_centroids(factory_string, nvecs)
    logger.info(f"Building index with factory string `{factory_string}`")

    if gpu_config is not None and torch.cuda.is_available():
        return build_gpu.build_faiss_index_multigpu(
            vectors,
            factory_string=factory_string,
            train_size=train_size,
            faiss_metric=faiss_metric,
            gpu_config=gpu_config,
        )

    return _build_faiss_index_on_cpu(
        vectors,
        factory_string=factory_string,
        train_size=train_size,
        faiss_metric=faiss_metric,
    )


def _build_faiss_index_on_cpu(
    vectors: dstruct.SizedDataset[np.ndarray],
    *,
    factory_string: str,
    train_size: Optional[int] = None,
    faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
) -> faiss.Index:
    vector_shape = vectors[0].shape
    vector_size = vector_shape[-1]
    index = faiss.index_factory(vector_size, factory_string, faiss_metric)

    if train_size is None:
        train_size = len(vectors)

    for i in range(0, len(vectors), train_size):
        batch = vectors[i : i + train_size]
        batch = np.asarray(batch).astype(np.float32)
        if i == 0:
            logger.info(f"Training faiss index on `{len(batch)}` vectors " f"({len(batch) / len(vectors):.2%} (cpu)")
            index.train(batch)  # type: ignore

        index.add(batch)

    if index.ntotal != len(vectors) or index.d != vector_size:
        raise ValueError(
            f"Index size doesn't match the size of the vectors."
            f"Found vectors: `{vector_size}`, index: `{index.ntotal, index.d}`"
        )

    return index
