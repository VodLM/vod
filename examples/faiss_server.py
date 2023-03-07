# pylint: disable=missing-function-docstring
import argparse
import tempfile
import timeit
from pathlib import Path

import faiss
import numpy as np
import rich
from loguru import logger

from raffle_ds_research.tools.index_tools.client import FaissClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_calls", type=int, default=1_000)
    parser.add_argument("--nprobe", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dataset_size", type=int, default=100_000)
    parser.add_argument("--vector_size", type=int, default=128)
    parser.add_argument("--index_factory", type=str, default="IVF100,Flat")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir, f"index.faiss")
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # generate some random data and fit and index, write it to disk.
        data = np.random.randn(args.dataset_size, args.vector_size).astype(np.float32)
        index = faiss.index_factory(
            args.vector_size,
            args.index_factory,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(data)
        index.add(data)
        faiss.write_index(index, str(index_path.absolute()))

        # Spawn a Faiss server and query it.
        log_dir = Path(tmpdir, "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        with FaissClient(
            index_path,
            log_dir=log_dir,
            nprobe=args.nprobe,
        ) as faiss_client:
            query_vec = data[: args.batch_size, :]
            search_results = faiss_client.search(query_vec, 3)
            rich.print(search_results)

            # time the API
            logger.info("Timing the API")
            timer = timeit.Timer(lambda: faiss_client.search(query_vec, 3))
            run_time = timer.timeit(number=args.n_calls)
            rich.print(f"API.search: {1000 * run_time / args.n_calls:.4f} ms/call")

            # time the base faiss
            index.nprobe = args.nprobe
            logger.info("Timing vanilla FAISS calls")
            timer = timeit.Timer(lambda: index.search(query_vec, 3))
            run_time = timer.timeit(number=args.n_calls)
            rich.print(f"FAISS.search: {1000 * run_time / args.n_calls:.4f} ms/call")
