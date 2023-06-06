import shutil
import tempfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import rich
from rich.progress import track
from rich.status import Status

from raffle_ds_research.tools import arguantic
from raffle_ds_research.tools.index_tools.faiss_tools import support
from raffle_ds_research.tools.index_tools.faiss_tools.build_gpu import FaissGpuConfig
from raffle_ds_research.tools.index_tools.index_factory import FaissFactoryConfig


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


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    nvecs: int = 10_000_000
    vdim: int = 512
    init_bs: int = 10_000
    train_size: int = 1_000_000
    centroids: str = "auto"
    nprobe: int = 32
    ncodes: int = 32
    nbits: int = 8
    nthreads: int = 16
    n_trials: int = 100
    bs: int = 100
    top_k: int = 100
    encoding: str = ""
    use_gpu: int = 1


if __name__ == "__main__":
    args = Args.parse()
    n_calls = 100
    nprobe = 1
    vec_path = Path("cache", "vecs", "vecs.npy")
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir, "cache/faiss")
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = Path(cache_dir, "index.faiss")

        rich.print(args)
        faiss.omp_set_num_threads(args.nthreads)

        factory_str = (
            f"IVF{args.centroids},PQ{args.ncodes}x{args.nbits}{args.encoding}"
            if args.ncodes > 0
            else f"IVF{args.centroids},Flat"
        )

        factory_str = support.infer_factory_centroids(factory_str, args.nvecs)

        config = FaissFactoryConfig(
            factory=factory_str,
            nprobe=args.nprobe,
            metric=faiss.METRIC_INNER_PRODUCT,
            train_size=None,
            gpu=FaissGpuConfig(use_float16=False, use_precomputed_tables=False) if args.use_gpu else None,
        )
        rich.print(config)

        # generate some random data and fit and index, write it to disk.
        if not vec_path.exists():
            with Status(f"Allocating vectors ({args.nvecs}, {args.vdim})", spinner="dots") as status:
                vectors = np.empty((args.nvecs, args.vdim), dtype=np.float32)
            for i in track(range(0, vectors.shape[0], args.init_bs), description="Setting random values.."):
                vslice = vectors[i : i + args.init_bs, :]
                vectors[i : i + args.init_bs, :] = np.random.randn(*vslice.shape)
            np.save(vec_path, vectors)
            del vectors
        vectors = np.memmap(vec_path, dtype=np.float32, mode="r", shape=(args.nvecs, args.vdim))
        vectors = LazyArrayLoader(vectors)
        rich.print(f"Vectors: {vectors.shape} ({type(vectors)}, {vectors.dtype})")

        index = faiss.index_factory(args.vdim, config.factory)
        rich.print(f"Index: {index}")

        if args.use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)

        x_train = vectors[: args.train_size, :] if args.train_size and args.train_size < args.nvecs else vectors[:]
        rich.print(f"Training on {x_train.shape} ({type(x_train)}, {x_train.dtype})")

        rich.print(f"Training on {x_train.shape}")
        index.train(x_train)

        rich.print(f"Adding {vectors.shape[0]} vectors to index")
        for i in track(range(0, vectors.shape[0], args.bs), description="Adding vectors.."):
            v_i = vectors[i : i + args.bs, :]
            index.add(v_i)

        # query = np.random.randn(args.bs, args.vdim).astype(np.float32)
        query = vectors[: args.bs, :]
        indices, scores = index.search(query, args.top_k)
        query = vectors[: args.bs, :]
        indices, scores = index.search(query, args.top_k)
        rich.print(f"Query: {query.shape} ({type(query)}, {query.dtype})")
        rich.print({"indices": indices, "scores": scores})
