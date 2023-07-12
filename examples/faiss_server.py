from __future__ import annotations

import tempfile
import timeit
from pathlib import Path

import faiss
import numpy as np
import rich
import torch
from loguru import logger

from src.vod_search import faiss_tools
from src.vod_tools import arguantic


class ProfileArgs(arguantic.Arguantic):
    """Arguments for the script."""

    n_calls: int = 1_000
    nprobe: int = 10
    batch_size: int = 10
    dataset_size: int = 1_000
    vector_size: tuple[int, ...] = (64,)
    index_factory: str = "IVF100,Flat"
    top_k: int = 10
    verbose: int = 0


def profile_faiss_server(arguments: ProfileArgs) -> dict[str, float]:
    """Show how to use the FaissClient."""
    benchmark = {}

    def _log_perf(run_time: float, n_calls: int, name: str) -> None:
        perf = 1000 * run_time / n_calls
        logger.info(f"{name}: {perf:.4f} ms/batch")
        benchmark[name] = perf

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir, "index.faiss")
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # generate some random data and fit and index, write it to disk.
        data = np.random.randn(arguments.dataset_size, arguments.vector_size).astype(np.float32)
        query_vec = data[: arguments.batch_size, :]
        index = faiss.index_factory(
            arguments.vector_size,
            arguments.index_factory,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(data)
        index.add(data)
        faiss.write_index(index, str(index_path.absolute()))

        # time the base faiss
        index.nprobe = arguments.nprobe
        logger.info("Timing in-thread `faiss.Index.search`")
        timer = timeit.Timer(lambda: index.search(query_vec, arguments.top_k))  # noqa: F821 -> ruff bug
        run_time = timer.timeit(number=arguments.n_calls)
        _log_perf(run_time, arguments.n_calls, "main_thread")
        del index

        # Spawn a Faiss server and query it.
        log_dir = Path(tmpdir, "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        with faiss_tools.FaissMaster(index_path, nprobe=arguments.nprobe, logging_level="critical") as faiss_master:
            faiss_client = faiss_master.get_client()
            # time the API
            logger.info("Timing the API (fast)")
            timer = timeit.Timer(lambda: faiss_client.search(vector=query_vec, top_k=arguments.top_k))
            run_time = timer.timeit(number=arguments.n_calls)
            _log_perf(run_time, arguments.n_calls, "API_fast")

            # time the API
            logger.info("Timing the API")
            timer = timeit.Timer(lambda: faiss_client.search_py(query_vec, arguments.top_k))
            run_time = timer.timeit(number=arguments.n_calls)
            _log_perf(run_time, arguments.n_calls, "API_base")

            if arguments.verbose:
                # show the results (numpy arrays)
                search_results = faiss_client.search(vector=query_vec[:3], top_k=arguments.top_k)
                rich.print(search_results)

                # show the results (torch tensors)
                search_results = faiss_client.search(vector=torch.from_numpy(query_vec[:3]), top_k=arguments.top_k)
                rich.print(search_results)

    return benchmark


if __name__ == "__main__":
    args = ProfileArgs.parse()
    profile_faiss_server(args)
