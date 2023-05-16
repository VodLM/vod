# pylint: disable=unused-import

from __future__ import annotations

import argparse
import re

import faiss
import numpy as np
import stackprinter
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger

from raffle_ds_research.tools.index_tools import io
from raffle_ds_research.tools.index_tools.faiss_tools import SearchFaissQuery
from raffle_ds_research.tools.index_tools.faiss_tools.build_gpu import FaissGpuConfig
from raffle_ds_research.tools.index_tools.faiss_tools.models import (
    FaissSearchResponse,
    FastFaissSearchResponse,
    FastSearchFaissQuery,
)
from raffle_ds_research.tools.index_tools.retrieval_data_type import RetrievalDataType
from raffle_ds_research.tools.utils.exceptions import dump_exceptions_to_file


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7678)
    parser.add_argument("--logging-level", type=str, default="INFO")
    parser.add_argument("--serve-on-gpu", action="store_true", default=False)
    return parser.parse_args()


def init_index(arguments: argparse.Namespace) -> faiss.Index:
    """Initialize the index."""
    logger.info("Initializing index")
    index = faiss.read_index(arguments.index_path)
    index.nprobe = arguments.nprobe
    return index


args = parse_args()
app = FastAPI()
logger.info("Starting API")
faiss_index = init_index(args)
if args.serve_on_gpu:
    logger.info("Using GPU")
    co = FaissGpuConfig().cloner_options()
    faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co=co)


@app.get("/")
def health_check() -> str:
    """Check if the index is ready."""
    if faiss_index.ntotal == 0:
        return "ERROR: Index is empty"
    if not faiss_index.is_trained:
        return "ERROR: Index is not trained"

    return "OK"


@app.post("/search")
async def search(query: SearchFaissQuery) -> FaissSearchResponse:
    """Search the index."""
    query_vec = np.array(query.vectors, dtype=np.float32)
    scores, indices = faiss_index.search(query_vec, k=query.top_k)
    return FaissSearchResponse(scores=scores.tolist(), indices=indices.tolist())


@dump_exceptions_to_file
@app.post("/fast-search")
async def fast_search(query: FastSearchFaissQuery) -> FastFaissSearchResponse:
    """Search the index."""
    try:
        deserializer = {
            RetrievalDataType.NUMPY: io.deserialize_np_array,
            RetrievalDataType.TORCH: io.deserialize_torch_tensor,
        }[query.array_type]
        query_vec = deserializer(query.vectors)
        if len(query_vec.shape) != 2:  # noqa: PLR2004
            raise ValueError(f"Expected 2D array, got {len(query_vec.shape)}D array")
        scores, indices = faiss_index.search(query_vec, k=query.top_k)
        return FastFaissSearchResponse(
            scores=io.serialize_np_array(scores),
            indices=io.serialize_np_array(indices),
        )
    except Exception as exc:
        trace = stackprinter.format()
        raise HTTPException(status_code=500, detail=str(trace)) from exc


def run_faiss_server(host: str = args.host, port: int = args.port) -> None:
    """Start the API."""
    pattern = re.compile(r"^(http|https)://")
    host = re.sub(pattern, "", host)
    uvicorn.run(app, host=host, port=port, workers=1, log_level=args.logging_level.lower())


if __name__ == "__main__":
    run_faiss_server()
