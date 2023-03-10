from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import faiss
import faiss.contrib.torch_utils
import numpy as np
import uvicorn
from fastapi import FastAPI
from loguru import logger

from raffle_ds_research.tools.index_tools import io
from raffle_ds_research.tools.index_tools.data_models import (
    FaissSearchResponse,
    FastFaissSearchResponse,
    FastSearchFaissQuery,
    SearchFaissQuery,
)
from raffle_ds_research.tools.index_tools.retrieval_data_type import RetrievalDataType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7678)
    parser.add_argument("--logging-level", type=str, default="INFO")
    parser.add_argument("--log-dir", type=str, default=None)
    return parser.parse_args()


def init_index(args: argparse.Namespace) -> faiss.Index:
    """Initialize the index"""
    logger.info("Initializing index")
    faiss_index = faiss.read_index(args.index_path)
    faiss_index.nprobe = args.nprobe
    return faiss_index


args = parse_args()
if args.log_dir is not None:
    logger.add(Path(args.log_dir, f"{os.getpid()}-faiss_server.log"), level=args.logging_level.upper())
app = FastAPI()
logger.info("Starting API")
faiss_index = init_index(args)


@app.get("/")
def health_check() -> str:
    """Check if the index is ready"""
    if faiss_index.ntotal == 0:
        return "ERROR: Index is empty"
    if not faiss_index.is_trained:
        return "ERROR: Index is not trained"

    return "OK"


@app.post("/search")
def search(query: SearchFaissQuery) -> FaissSearchResponse:
    """Search the index"""
    query_vec = np.array(query.vectors, dtype=np.float32)
    scores, indices = faiss_index.search(query_vec, k=query.top_k)
    return FaissSearchResponse(scores=scores.tolist(), indices=indices.tolist())


@app.post("/fast-search")
def fast_search(query: FastSearchFaissQuery) -> FastFaissSearchResponse:
    """Search the index. TODO: use gRPC to speed up communication."""
    deserializer = {
        RetrievalDataType.NUMPY: io.deserialize_np_array,
        RetrievalDataType.TORCH: io.deserialize_torch_tensor,
    }[query.array_type]
    query_vec = deserializer(query.vectors)
    if len(query_vec.shape) != 2:
        raise ValueError(f"Expected 2D array, got {len(query_vec.shape)}D array")
    scores, indices = faiss_index.search(query_vec, k=query.top_k)
    return FastFaissSearchResponse(
        scores=io.serialize_np_array(scores),
        indices=io.serialize_np_array(indices),
    )


def run_faiss_server(host: str = args.host, port: int = args.port):
    """Start the API"""
    pattern = re.compile(r"^(http|https)://")
    host = re.sub(pattern, "", host)
    uvicorn.run(app, host=host, port=port, workers=1, log_level=args.logging_level.lower())


if __name__ == "__main__":
    run_faiss_server()
