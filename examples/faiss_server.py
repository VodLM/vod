# pylint: disable=missing-function-docstring
import argparse
import dataclasses
import tempfile
import timeit
from pathlib import Path

import faiss
import numpy as np
import rich
import torch
from loguru import logger

from raffle_ds_research.tools.index_tools.client import FaissMaster


@dataclasses.dataclass()
class ProfileArgs:
    n_calls: int = 1_000
    nprobe: int = 10
    batch_size: int = 10
    dataset_size: int = 1_000
    vector_size: int = (64,)
    index_factory: str = "IVF100,Flat"
    top_k: int = 10
    verbose: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(**vars(args))


def parse_args() -> ProfileArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_calls", type=int, default=1_000)
    parser.add_argument("--nprobe", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dataset_size", type=int, default=100_000)
    parser.add_argument("--vector_size", type=int, default=512)
    parser.add_argument("--index_factory", type=str, default="Flat")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--verbose", type=bool, default=True)
    return ProfileArgs.from_args(parser.parse_args())


def profile_faiss_server(args: ProfileArgs) -> dict[str, float]:
    """Show how to use the FaissClient."""
    benchmark = {}

    def _log_perf(run_time: float, n_calls: int, name: str):
        perf = 1000 * run_time / n_calls
        logger.info(f"{name}: {perf:.4f} ms/batch")
        benchmark[name] = perf

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir, f"index.faiss")
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # generate some random data and fit and index, write it to disk.
        data = np.random.randn(args.dataset_size, args.vector_size).astype(np.float32)
        query_vec = data[: args.batch_size, :]
        index = faiss.index_factory(
            args.vector_size,
            args.index_factory,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(data)
        index.add(data)
        faiss.write_index(index, str(index_path.absolute()))

        # time the base faiss
        index.nprobe = args.nprobe
        logger.info("Timing in-thread `faiss.Index.search`")
        timer = timeit.Timer(lambda: index.search(query_vec, args.top_k))
        run_time = timer.timeit(number=args.n_calls)
        _log_perf(run_time, args.n_calls, "main_thread")
        del index

        # Spawn a Faiss server and query it.
        log_dir = Path(tmpdir, "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        with FaissMaster(index_path, log_dir=log_dir, nprobe=args.nprobe, logging_level="critical") as faiss_master:
            faiss_client = faiss_master.get_client()

            # time the API
            logger.info("Timing the API (fast)")
            timer = timeit.Timer(lambda: faiss_client.search(query_vec, args.top_k))
            run_time = timer.timeit(number=args.n_calls)
            _log_perf(run_time, args.n_calls, "API_fast")

            # time the API
            logger.info("Timing the API")
            timer = timeit.Timer(lambda: faiss_client.search_py(query_vec, args.top_k))
            run_time = timer.timeit(number=args.n_calls)
            _log_perf(run_time, args.n_calls, "API_base")

            if args.verbose:
                # show the results (numpy arrays)
                search_results = faiss_client.search(query_vec[:3], args.top_k)
                rich.print(search_results)

                # show the results (torch tensors)
                search_results = faiss_client.search(torch.from_numpy(query_vec[:3]), args.top_k)
                rich.print(search_results)

    return benchmark


if __name__ == "__main__":
    args = parse_args()
    profile_faiss_server(args)
