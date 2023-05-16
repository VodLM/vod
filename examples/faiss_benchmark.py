from __future__ import annotations

import collections
import dataclasses
import datetime
import json
import os
import pathlib
import shutil
import tarfile
import time
from contextlib import closing
from typing import Any, Callable, Optional
from urllib import request

import datasets
import faiss
import numpy as np
import psutil
import pydantic
import rich
import rich.status
import torch
from loguru import logger
from rich.progress import track

from raffle_ds_research.tools import arguantic


def _get_memory_usage() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3


class CustomEncoder(json.JSONEncoder):
    """Custom encoder to handle special types."""

    def default(self, obj: object) -> Any:  # noqa: ANN401
        """Default encoder."""
        if isinstance(obj, pathlib.Path):
            return str(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


TEXMEX_DATASETS = {
    "siftsmall": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
    "sift": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
    "gist": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
}


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    dset: str = "sift"
    cache_dir: pathlib.Path = "~/.raffle/datasets/vectors-bases/"
    output_dir: pathlib.Path = "faiss-index-benchmark/"
    repeat: int = 0
    expand: int = 0
    noise: float = 1.0
    factory: str = "Flat"
    n_train: int = -1
    n_probe: int = 16
    batch_size: int = 100
    top_k: int = 300
    n_samples: int = -1
    omp_num_threads: int = 32

    @pydantic.validator("cache_dir", pre=True, always=True, allow_reuse=True)  # type: ignore
    @pydantic.validator("output_dir", pre=True, always=True, allow_reuse=True)
    def _validate_dir(cls, v: str | pathlib.Path) -> pathlib.Path:  # noqa: N805
        return pathlib.Path(v).expanduser().resolve()

    @property
    def dset_url(self) -> str:
        """Return the URL where to download the dataset."""
        return TEXMEX_DATASETS[self.dset]

    @property
    def fingerprint(self) -> str:
        """Return a fingerprint of the arguments."""
        exclude = ["cache_dir", "output_dir"]
        args = self.dict()
        for key in exclude:
            args.pop(key)

        return datasets.fingerprint.Hasher().hash(json.dumps(args))


class WithTimer:
    """Context manager to time a block of code."""

    _status: Optional[rich.status.Status] = None

    def __init__(
        self,
        store: dict[str, float],
        key: str,
        log_fn: Optional[Callable[[Any], Any]] = logger.info,
        status: bool = False,
    ) -> None:
        self.store = store
        self.key = key
        self.log_fn = log_fn
        self.use_status = status

    def __enter__(self) -> None:
        """Start the timer."""
        self.start = time.time()
        if self.use_status:
            self._status = rich.status.Status(f"{self.key}...")
            self._status.__enter__()

    def __exit__(self, *args) -> None:  # noqa: ANN
        """Stop the timer and log the time."""
        self.store[self.key] = time.time() - self.start
        if self.log_fn is not None:
            self.log_fn(f"{self.key}: {self.store[self.key]:.2f}s")
        if self._status is not None:
            self._status.__exit__(*args)
            self._status = None


# now define a function to read the fvecs file format of Sift1M dataset
def _read_fvecs(fp: pathlib.Path, dtype: str = "float32") -> np.ndarray:
    a = np.fromfile(fp, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32").astype(dtype)


def _read_ivecs(fp: pathlib.Path, dtype: str = "int32") -> np.ndarray:
    a = np.fromfile(fp, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].astype(dtype)


def _repr_arr(x: np.ndarray) -> str:
    return f"np.ndarray(shape={x.shape}, dtype={x.dtype})"


@dataclasses.dataclass(frozen=True)
class VectorDataset:
    """Vector dataset."""

    vectors: np.ndarray
    labels: np.ndarray
    queries: np.ndarray

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"{type(self).__name__}("
            f"vectors={_repr_arr(self.vectors)}, "
            f"labels={_repr_arr(self.labels)}, "
            f"queries={_repr_arr(self.queries)})"
        )

    def repeat(self, n: int, noise: float = 1e-0) -> VectorDataset:
        """Repeat the dataset n times."""
        if n < 1:
            return self

        new_vectors = np.repeat(self.vectors[None], n, axis=0)
        new_vectors = new_vectors.reshape(n * self.vectors.shape[0], self.vectors.shape[1])
        if noise > 0:
            new_vectors += np.random.randn(*new_vectors.shape) * noise
        # add labels for the repeated vectors, offseting by len(self.vectors) for each repetition

        new_labels = np.repeat(self.labels[:, None, :], n, axis=1)
        offset = len(self.vectors) * np.arange(n, dtype=self.labels.dtype)
        new_labels += offset[None, :, None]
        new_labels = new_labels.reshape(self.labels.shape[0], n * self.labels.shape[1])

        return VectorDataset(
            vectors=new_vectors,
            labels=new_labels,
            queries=self.queries,
        )

    def expand(self, n: int, noise: float = 1e-0) -> VectorDataset:
        """Expand the vector dimension."""
        if n < 1:
            return self

        def _expand_vector_dim(x: np.ndarray, noise: float) -> np.ndarray:
            new_x = np.repeat(x, n, axis=1)
            if noise > 0:
                new_x += np.random.randn(*new_x.shape) * noise
            return new_x

        return VectorDataset(
            vectors=_expand_vector_dim(self.vectors, noise),
            queries=_expand_vector_dim(self.queries, noise),
            labels=self.labels,
        )


def _load_knn_search_dataset(args: Args) -> VectorDataset:
    """Load a dataset for approximate nearest neighbor search.

    See http://corpus-texmex.irisa.fr/
    """
    fname = pathlib.Path(args.dset_url).name
    zip_path = args.cache_dir / fname
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    fpath = zip_path.parent / args.dset
    if not fpath.exists():
        logger.info(f"Downloading `{args.dset}` dataset from `{args.dset_url}`")
        with closing(request.urlopen(args.dset_url)) as r, zip_path.open("wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f)

        tar = tarfile.open(zip_path, "r:gz")
        tar.extractall(path=args.cache_dir)

    logger.info(f"Loading `{args.dset}` dataset from `{fpath}`")
    wb = _read_fvecs(fpath / f"{args.dset}_base.fvecs", dtype="float32")  # 1M samples
    xq = _read_fvecs(fpath / f"{args.dset}_query.fvecs", dtype="float32")  # 10K queries
    yb = _read_ivecs(fpath / f"{args.dset}_groundtruth.ivecs", dtype="int64")  # 10K queries

    return VectorDataset(vectors=wb, queries=xq, labels=yb)


def run(args: Args) -> None:
    """Benchmark faiss indexes."""
    mem_init = _get_memory_usage()
    dset = _load_knn_search_dataset(args)
    dset = dset.repeat(args.repeat, noise=args.noise)
    dset = dset.expand(args.expand, noise=args.noise)
    rich.print(dset)
    runtimes = {}

    faiss.omp_set_num_threads(args.omp_num_threads)
    index = faiss.index_factory(dset.vectors.shape[1], args.factory, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = args.n_probe

    # train the index
    with WithTimer(store=runtimes, key="training", status=True):
        _train_index(index, dset, args=args)

    # add vectors to the index
    with WithTimer(store=runtimes, key="populating", status=True):
        _populate_index(index, dset)

    # report the memory usage
    mem_loaded = _get_memory_usage()
    logger.info(f"Memory usage: {mem_loaded - mem_init:.3f} GB")

    # benchmark the index
    with WithTimer(store=runtimes, key="benchmarking"):
        benchmark = _benchmark_index(index, dset, args=args)

    report = {
        "date": str(datetime.datetime.now(tz=datetime.timezone.utc).isoformat()),
        "args": args.dict(),
        "runtimes": runtimes,
        "benchmark": benchmark,
    }
    rich.print(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{args.fingerprint}.json"
    with output_file.open("w") as f:
        json.dump(report, f, indent=2, cls=CustomEncoder)


def _compute_metrics(scores: np.ndarray, indices: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = torch.from_numpy(scores)
    indices = torch.from_numpy(indices)
    labels = torch.from_numpy(labels)
    is_match = (indices[:, :, None] == labels[:, None, :]).any(dim=2)
    sorted_is_match = is_match.gather(dim=1, index=scores.argsort(dim=1, descending=True))
    metrics = {}
    for k in [1, 10, 100]:
        metrics[f"r@{k}"] = (sorted_is_match[:, :k].sum(dim=1).float() / labels.shape[1]).mean()
    for k in [1, 10, 100]:
        metrics[f"p@{k}"] = (sorted_is_match[:, :k].float().mean(dim=1)).mean()
    for k in [1, 10, 100]:
        metrics[f"hit@{k}"] = sorted_is_match.any(dim=1).float().mean()
    return metrics


def _benchmark_index(index: faiss.Index, dset: VectorDataset, *, args: Args) -> dict[str, float]:
    metrics = collections.defaultdict(list)
    n_total = len(dset.queries) if args.n_samples < 0 else args.n_samples
    for i in track(
        range(0, n_total, args.batch_size), description=f"Benchmarking (total={n_total}, top_k={args.top_k})"
    ):
        xq = dset.queries[i : i + args.batch_size]
        yq = dset.labels[i : i + args.batch_size]

        scores, indices = index.search(xq, args.top_k)
        for k, v in _compute_metrics(scores, indices, yq).items():
            metrics[k].append(v)

    return {k: np.mean(v) for k, v in metrics.items()}


def _populate_index(index: faiss.Index, dset: VectorDataset) -> None:
    index.add(dset.vectors)


def _train_index(index: faiss.Inde, dset: VectorDataset, *, args: Args) -> None:
    x_train = dset.vectors[: args.n_train] if args.n_train > 0 else dset.vectors
    index.train(x_train)


if __name__ == "__main__":
    args = Args.parse()
    run(args)
