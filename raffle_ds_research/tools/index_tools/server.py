from __future__ import annotations

import argparse

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI
from loguru import logger

from raffle_ds_research.tools.index_tools.data_models import FaissSearchResults, SearchFaissQuery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7678)
    return parser.parse_args()


def init_index(args: argparse.Namespace) -> faiss.Index:
    """Initialize the index"""
    logger.info("Initializing index")
    faiss_index = faiss.read_index(args.index_path)
    # faiss_index.nprobe = args.nprobe
    return faiss_index


args = parse_args()
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


@app.post(
    "/search",
)
def search(query: SearchFaissQuery) -> FaissSearchResults:
    """Search the index"""
    query_vec = np.array(query.vectors, dtype=np.float32)
    scores, indices = faiss_index.search(query_vec, k=query.top_k)
    return FaissSearchResults(scores=scores.tolist(), indices=indices.tolist())


def run_faiss_server(host: str = args.host, port: int = args.port):
    """Start the API"""
    uvicorn.run(app, host=host, port=port, workers=1, log_level="critical")


if __name__ == "__main__":
    run_faiss_server()
