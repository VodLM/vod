from __future__ import annotations

import tempfile
import time

import faiss
import numpy as np
import rich
from loguru import logger
from rich.progress import track
from vod_search import faiss_search
from vod_tools import arguantic


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    dataset_size: int = 3_000
    n_categories: int = 100
    vector_size: int = 128
    batch_size: int = 10
    top_k: int = 100
    n_trials: int = 10


def run(args: Args) -> None:
    """Run the script."""
    vectors = np.random.randn(args.dataset_size, args.vector_size).astype("float32")

    # Build a faiss index
    index = faiss_search.build_faiss_index(vectors=vectors, factory_string="IVFauto,Flat")  # type: ignore

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = f"{tmpdir}/index.faiss"
        faiss.write_index(index, index_path)

        # Spin up a server
        with faiss_search.FaissMaster(index_path) as master:
            client = master.get_client()
            rich.print(client)

            query_vecs = np.random.randn(args.batch_size, args.vector_size).astype("float32")
            query_groups = np.random.randint(0, args.n_categories, size=args.batch_size).astype("int64")

            results = client.search(
                vector=query_vecs,
                group=query_groups,  # type: ignore
                top_k=3,
            )
            rich.print(
                {
                    "search_results": results,
                }
            )

            # Benchmark
            logger.info("Benchmarking...")
            start = time.perf_counter()
            for _ in track(range(args.n_trials), description="Benchmarking Qdrant"):
                query_vecs = np.random.randn(args.batch_size, args.vector_size).astype("float32")
                results = client.search(
                    vector=query_vecs,
                    top_k=args.top_k,
                )
            end = time.perf_counter()
            logger.info(f"Qdrant: {1000*(end - start) / args.n_trials:.3f} ms/batch")


if __name__ == "__main__":
    args = Args.parse()
    rich.print(args)
    run(args)
