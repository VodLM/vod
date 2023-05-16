import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import psutil
import rich
import torch
from loguru import logger
from rich.progress import track
from rich.status import Status

from raffle_ds_research.tools import arguantic, index_tools
from raffle_ds_research.tools.index_tools.faiss_tools.build_gpu import FaissGpuConfig
from raffle_ds_research.tools.index_tools.index_factory import FaissFactoryConfig


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    nvecs: int = 10_000_000
    vdim: int = 512
    init_bs: int = 10_000
    train_size: int = 1_000_000
    prep: Optional[str] = None
    factory: str = "OPQ32,IVFauto,PQ32x8"
    nprobe: int = 32
    nthreads: int = 16
    n_trials: int = 100
    bs: int = 100
    top_k: int = 100
    use_gpu: bool = True
    pre_tables: bool = False
    use_float16: bool = True
    verbose: bool = True
    serve_on_gpu: bool = False


def _get_cuda_device_info() -> dict[str, Any]:
    def _get_device_info() -> str:
        line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
        line = line_as_bytes.decode("ascii")
        _, line = line.split(":", 1)
        line, _ = line.split("(")
        return line.strip()

    return {
        "device": _get_device_info(),
        "n_devices": torch.cuda.device_count(),
        "version": torch.version.cuda,
    }


def _get_available_memory() -> float:
    """Return the available RAM in GB."""
    return psutil.virtual_memory().available / (1024**3)


class LazyArrayLoader:
    """Lazy loader for mem-mapped numpy arrays."""

    def __init__(self, array: np.memmap) -> None:
        self.array = array

    def __getitem__(self, index: Any) -> np.ndarray:  # noqa: ANN401
        return np.asarray(self.array[index])

    def __len__(self) -> int:
        return len(self.array)

    @property
    def shape(self) -> tuple:
        """Return the shape of the array."""
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the array."""
        return self.array.dtype


def build_and_eval_index(args: Args) -> dict[str, Any]:
    """Build a faiss index on GPU and serve it."""
    if args.verbose:
        rich.print(args)
    times = {}
    vec_path = Path("cache", "vecs", "vecs.npy")
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir, "cache/faiss")
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        faiss.omp_set_num_threads(args.nthreads)
        config = FaissFactoryConfig(
            factory=args.factory,
            nprobe=args.nprobe,
            metric=faiss.METRIC_INNER_PRODUCT,
            train_size=args.train_size,
            gpu=FaissGpuConfig(
                use_precomputed_tables=bool(args.pre_tables),
                use_float16=bool(args.use_float16),
            )
            if args.use_gpu
            else None,
        )
        if args.verbose:
            rich.print(config)

        # generate some random data and fit and index, write it to disk.
        if not vec_path.exists():
            with Status(f"Allocating vectors ({args.nvecs}, {args.vdim})", spinner="dots"):
                vectors = np.empty((args.nvecs, args.vdim), dtype=np.float32)
            for i in track(range(0, vectors.shape[0], args.init_bs), description="Setting random values.."):
                vslice = vectors[i : i + args.init_bs, :]
                vectors[i : i + args.init_bs, :] = np.random.randn(*vslice.shape)
            np.save(vec_path, vectors)
            del vectors
        vectors = np.memmap(vec_path, dtype=np.float32, mode="r", shape=(args.nvecs, args.vdim))
        vectors = LazyArrayLoader(vectors)
        logger.info(f"Vectors: {vectors.shape}")

        t_0 = time.time()
        _pre_mem = _get_available_memory()
        logger.info(f"Avaliable memory: {_pre_mem:.3f} GB")
        with index_tools.build_faiss_index(
            vectors=vectors,  # type: ignore
            config=config,
            cache_dir=cache_dir,
            skip_setup=False,
            barrier_fn=None,
            serve_on_gpu=args.serve_on_gpu,
        ) as master:
            times["build"] = time.time() - t_0
            mem_usage = _pre_mem - _get_available_memory()
            logger.info(f"Memory usage: {mem_usage:.3f} GB")
            faiss_client = master.get_client()
            query_vec = vectors[:10, :]
            search_results = faiss_client.search(vector=query_vec, top_k=10)
            if args.verbose:
                rich.print(search_results)

            # time the API
            logger.info(f"Timing API calls (n_trials={args.n_trials})")
            t_0 = time.time()
            for _ in range(args.n_trials):
                query = np.random.randn(args.bs, args.vdim).astype(np.float32)
                faiss_client.search(vector=query, top_k=args.top_k)
            t_1 = time.time()
            times["search"] = (t_1 - t_0) / args.n_trials
            logger.info(f"API call time: {1000*times['search']:.3f} ms/batch")

    return {"times": times, "args": args.dict(), "cuda": _get_cuda_device_info(), "mem_usage": mem_usage}


if __name__ == "__main__":
    args = Args.parse()
    info = build_and_eval_index(args)
    rich.print(info)
