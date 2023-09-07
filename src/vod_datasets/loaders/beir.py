"""Credits to `github.com/beir-cellar/beir`."""

import collections
import pathlib
import zipfile
from typing import Any

import datasets
import requests
import rich
from loguru import logger
from tqdm import tqdm
from typing_extensions import Self, Type
from vod_configs.py.datasets import DatasetLoader
from vod_datasets.rosetta import models


def _download_url(
    url: str,
    save_path: str | pathlib.Path,
    chunk_size: int = 1024,
) -> pathlib.Path:
    r = requests.get(url, stream=True, timeout=300)
    total = int(r.headers.get("Content-Length", 0))
    with open(save_path, "wb") as fd, tqdm(
        desc=str(save_path),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)

    return pathlib.Path(save_path)


def _unzip(zip_file: str | pathlib.Path, out_dir: str | pathlib.Path) -> pathlib.Path:
    with zipfile.ZipFile(zip_file, "r") as zip_:
        zip_.extractall(path=out_dir)
    return pathlib.Path(out_dir)


def _download_and_unzip(url: str, out_dir: str | pathlib.Path, chunk_size: int = 1024) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = url.split("/")[-1]
    zip_file = out_dir / dataset_name

    if not zip_file.is_file():
        logger.info("Downloading {dataset_name} ...", dataset_name=dataset_name)
        _download_url(url, zip_file, chunk_size)

    if not (out_dir / dataset_name.replace(".zip", "")).is_dir():
        logger.info("Unzipping {dataset_name} ...", dataset_name=dataset_name)
        _unzip(zip_file, out_dir)

    return out_dir / dataset_name.replace(".zip", "")


def _validate_file(path: str | pathlib.Path, ext: str) -> None:
    """Validate if the file is present and has the correct extension."""
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError("File `{}` not present!".format(path))

    if not path.name.endswith(ext):
        raise ValueError("File `{}` must be present with extension `{}`".format(path, ext))


class QrelsDataset:
    """Qrels dataset."""

    queries: datasets.Dataset
    qrels: datasets.Dataset
    corpus: datasets.Dataset

    _queries_columns: set[str] = {"id", "text"}
    _qrels_columns: set[str] = {"query-id", "corpus-id", "score"}
    _corpus_columns: set[str] = {"_id", "title", "text"}

    def __init__(self, queries: datasets.Dataset, qrels: datasets.Dataset, corpus: datasets.Dataset):
        self.queries = queries
        self.qrels = qrels
        self.corpus = corpus
        self._validate()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  queries={self.queries},\n  qrels={self.qrels},\n  corpus={self.corpus})"

    def _validate(self) -> None:
        if self._queries_columns > set(self.queries.column_names):
            raise ValueError(f"Queries columns `{self._queries_columns}` not present in `{self.queries.column_names}`.")
        if self._qrels_columns > set(self.qrels.column_names):
            raise ValueError(f"Qrels columns `{self._qrels_columns}` not present in `{self.qrels.column_names}`.")
        if self._corpus_columns > set(self.corpus.column_names):
            raise ValueError(f"Corpus columns `{self._corpus_columns}` not present in `{self.corpus.column_names}`.")

    @classmethod
    def from_files(
        cls: Type[Self],
        queries: str | pathlib.Path,
        qrels: str | pathlib.Path,
        corpus: str | pathlib.Path,
    ) -> Self:
        """Load the dataset from files."""
        qrelss = sorted(pathlib.Path(qrels).parent.glob(pathlib.Path(qrels).name)) if "*" in str(qrels) else [qrels]
        _validate_file(queries, ext=".jsonl")
        for qrels in qrelss:
            _validate_file(qrels, ext=".tsv")
        _validate_file(corpus, ext=".jsonl")

        # Load the `qrels` parts
        qrels_parts = [
            datasets.load_dataset("csv", data_files=str(qrels), delimiter="\t")["train"]  # type: ignore
            for qrels in qrelss
        ]

        # Load the full dataset
        return cls(
            queries=datasets.load_dataset("json", data_files=str(queries))["train"],  # type: ignore
            qrels=datasets.concatenate_datasets(qrels_parts),  # type: ignore
            corpus=datasets.load_dataset("json", data_files=str(corpus))["train"],  # type: ignore
        )


class BeirDatasetLoader(DatasetLoader):
    """This class is an adaptation from `beir-cellar/beir`.

    `https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L1C1-L118C30`.
    """

    BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
    DATASETS = [
        "msmarco",
        "mmarco",
        "trec-covid",
        "nfcorpus",
        "nq",
        "hotpotqa",
        "fiqa",
        "arguana",
        "webis-touche2020",
        "cqadupstack",
        "quora",
        "dbpedia-entity",
        "scidocs",
        "fever",
        "climate-fever",
        "scifact",
        "germanquad",
    ]

    def __init__(  # noqa: PLR0913
        self,
        cache_dir: None | str = None,
        what: models.DatasetType = "queries",
    ):
        self.what = what
        self.cache_dir = pathlib.Path(cache_dir or models.DATASETS_CACHE_PATH) / "beir_datasets"

    def __call__(self, subset: str | None = None, split: str | None = None, **kwargs: Any) -> datasets.Dataset:
        """Load the dataset."""
        if subset not in self.DATASETS:
            raise ValueError(f"susbet `{subset}` not available. Please choose from `{self.DATASETS}`.")

        dataset_dir = _download_and_unzip(
            self.BASE_URL.format(subset),
            self.cache_dir,
        )

        # Load the data
        data = QrelsDataset.from_files(
            queries=dataset_dir / "queries.jsonl",
            qrels=dataset_dir / "qrels" / f"{split or '*'}.tsv",
            corpus=dataset_dir / "corpus.jsonl",
        )

        rich.print(data)
        rich.print(
            {
                "queries": data.queries[0],
                "qrels": data.qrels[0],
                "corpus": data.corpus[0],
            }
        )

        rich.print(collections.Counter(data.qrels["score"]))

        raise NotImplementedError
