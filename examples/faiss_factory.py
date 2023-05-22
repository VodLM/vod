import functools
import multiprocessing as mp
import os
import re
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

from raffle_ds_research.tools import arguantic, dstruct, index_tools
from raffle_ds_research.tools.index_tools.faiss_tools.build_gpu import FaissGpuConfig
from raffle_ds_research.tools.index_tools.index_factory import FaissFactoryConfig


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    nvecs: int = 100_000
    vdim: int = 512
    init_bs: int = 1_000
    train_size: Optional[int] = None
    prep: Optional[str] = None
    factory: str = "OPQ32,IVFauto,PQ32x8"
    nprobe: int = 32
    nthreads: Optional[int] = None
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
        content_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
        content = content_as_bytes.decode("ascii")
        ptrn = re.compile(r"GPU (\d+): (?P<device>.*) \(UUID: (?P<uuid>.*)")
        device_info = ptrn.search(content)
        if device_info is None:
            raise RuntimeError(f"Failed to parse `nvidia-smi -L` content={content}")
        return device_info.groupdict()["device"]

    return {
        "device": _get_device_info(),
        "n_devices": torch.cuda.device_count(),
        "version": torch.version.cuda,
    }


def _get_available_memory() -> float:
    """Return the available RAM in GB."""
    return psutil.virtual_memory().available / (1024**3)


def build_and_eval_index(args: Args) -> dict[str, Any]:
    """Build a faiss index on GPU and serve it."""
    if args.verbose:
        rich.print(args)
    times = {}
    vec_path = Path("cache", "vecs", f"vecs-{args.nvecs}x{args.vdim}.ts").absolute()
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir, "cache/faiss")
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if args.nthreads is not None:
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

        # generate some random data.
        if not vec_path.exists():
            _write_store(args, vec_path)

        store = dstruct.TensorStoreFactory.from_path(vec_path)
        vectors = dstruct.as_lazy_array(store)
        # vectors = np.random.randn(args.nvecs, args.vdim).astype(np.float32)
        logger.info(f"Vectors: {vectors.shape}")
        if args.verbose:
            rich.print(vectors[:])

        with Status("Waiting to free up memory..."):
            time.sleep(1)  # <-- this is a hack to get the memory usage to be more accurate
        _pre_mem = _get_available_memory()
        t_0 = time.time()
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
            for _ in track(range(args.n_trials), description=f"{args.n_trials} API calls ({args.bs}x{args.top_k})"):
                query = np.random.randn(args.bs, args.vdim).astype(np.float32)
                faiss_client.search(vector=query, top_k=args.top_k)
            t_1 = time.time()
            times["search"] = (t_1 - t_0) / args.n_trials
            logger.info(f"API call time: {1000*times['search']:.3f} ms/batch")

    return {"times": times, "args": args.dict(), "cuda": _get_cuda_device_info(), "mem_usage": mem_usage}


def _write_store(args: Args, vec_path: Path) -> None:
    store = dstruct.TensorStoreFactory.instantiate(vec_path, shape=(args.nvecs, args.vdim))
    vstore = store.open(create=True, delete_existing=False)
    gn_values_fn = functools.partial(_gen_random_values, shape=vstore.shape)
    futures = []
    buffer_size = 10
    with mp.Pool(processes=max(1, os.cpu_count() - 2)) as pool:
        ids_chunks = (list(range(i, min(i + args.init_bs, args.nvecs))) for i in range(0, args.nvecs, args.init_bs))
        for j, (ids, vec) in enumerate(
            track(
                pool.imap_unordered(gn_values_fn, ids_chunks),
                total=args.nvecs // args.init_bs,
                description="Generating random vectors",
            )
        ):
            f = vstore[ids].write(vec)
            futures.append(f)

            # start consumming the queue
            if j > buffer_size:
                f = futures.pop(0)
                f.result()

                # wait for the rest of the futures to finish
        for f in futures:
            f.result()

    if not vec_path.exists():
        raise RuntimeError(f"Failed to write vectors to {vec_path}")


def _gen_random_values(ids: list[int], *, shape: tuple[int, ...]) -> tuple[list[int], np.ndarray]:
    rgn = np.random.RandomState(ids[0])
    return ids, rgn.randn(len(ids), shape[1]).astype(np.float32)


if __name__ == "__main__":
    args = Args.parse()
    info = build_and_eval_index(args)
    rich.print(info)
