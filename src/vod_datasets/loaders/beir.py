"""Credits to `github.com/beir-cellar/beir`."""

import collections
import pathlib
import zipfile
from typing import Any

import datasets
import requests
from datasets import fingerprint
from loguru import logger
from tqdm import tqdm
from typing_extensions import Self, Type
from vod_configs.datasets import DatasetLoader
from vod_datasets.rosetta import models
from vod_datasets.rosetta.adapters.rename_fields import RenameSectionAdapter


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
        avail_files = [f.name for f in path.parent.iterdir()]
        raise ValueError(f"File `{path}` doesn't exist. Available files in `{path.parent}`: `{avail_files}`")

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
    SUBSETS = [
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

    def __init__(
        self,
        cache_dir: None | str = None,
        num_proc: int = 4,
        what: models.DatasetType = "queries",
    ):
        self.num_proc = num_proc
        self.what = what
        self.cache_dir = pathlib.Path(cache_dir or models.DATASETS_CACHE_PATH) / "beir_datasets"

    def __call__(self, subset: str | None = None, split: str | None = None, **kwargs: Any) -> datasets.Dataset:
        """Load the dataset."""
        if subset is None:
            raise ValueError("Please specify a subset.")
        if subset.startswith("hf://"):
            data = self._download_from_hf(subset.replace("hf://", ""), split)
        else:
            data = self._download_from_tu_darmstadt(subset, split)

        if self.what == "sections":
            return RenameSectionAdapter.translate_dset(data.corpus)
        if self.what == "queries":
            data.queries.cleanup_cache_files()
            output = data.queries.map(
                _FilterAndAssignRetrievalIds(data.qrels),
                num_proc=self.num_proc,
                remove_columns=data.queries.column_names,
                batched=True,
                desc=f"{subset}:{split or 'all'} - Formatting and filtering queries",
            )
            if len(output) == 0:
                raise ValueError(f"Dataset `{subset}` is empty.")
            return output

        raise ValueError(f"Unknown dataset type `{self.what}`.")

    def _download_from_hf(self, path_or_name: str, split: str | None = None) -> QrelsDataset:
        qrels = datasets.load_dataset(f"{path_or_name}-qrels", split=split)
        if isinstance(qrels, datasets.DatasetDict):
            qrels = datasets.concatenate_datasets([qrels[k] for k in sorted(qrels)])

        return QrelsDataset(
            qrels=qrels,  # type: ignore
            queries=datasets.load_dataset(path_or_name, "queries", split="queries"),  # type: ignore
            corpus=datasets.load_dataset(path_or_name, "corpus", split="corpus"),  # type: ignore
        )

    def _download_from_tu_darmstadt(self, subset: str, split: str | None = None) -> QrelsDataset:
        name, *subpath = str(subset).split("/")
        if name not in self.SUBSETS:
            raise ValueError(f"susbet `{name}` not available. Please choose from `{self.SUBSETS}`.")

        # Download the dataset
        dataset_dir = _download_and_unzip(
            self.BASE_URL.format(name),
            self.cache_dir,
        )

        # Load the data
        return QrelsDataset.from_files(
            queries=pathlib.Path(dataset_dir, *subpath, "queries.jsonl"),
            qrels=pathlib.Path(dataset_dir, *subpath, "qrels", f"{split or '*'}.tsv"),
            corpus=pathlib.Path(dataset_dir, *subpath, "corpus.jsonl"),
        )


class _FilterAndAssignRetrievalIds:
    output_model = models.QueryModel
    _lookup: dict[int, list[int]]

    def __init__(self, qrels: datasets.Dataset) -> None:
        self.qrels = qrels
        self._lookup: dict[int, list[int]] = {}

    @property
    def lookup(self) -> dict[int, list[int]]:
        """Build the lookup table lazily in each worker.

        This allows `datasets.map()` to cache this operation without building the lookup table.
        """
        if len(self._lookup) == 0:
            for row in self.qrels:
                qid = int(row["query-id"])  # type: ignore
                cid = int(row["corpus-id"])  # type: ignore
                score = float(row["score"])  # type: ignore
                if score <= 0:
                    continue
                if qid not in self._lookup:
                    self._lookup[qid] = [cid]
                elif cid not in self._lookup[qid]:
                    self._lookup[qid].append(cid)

        return self._lookup

    def __call__(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output = collections.defaultdict(list)
        for qid, txt in zip(batch["_id"], batch["text"]):
            qid_int = int(qid)
            retrieval_ids = self.lookup.get(qid_int, None)
            if retrieval_ids is None:
                continue
            model = self.output_model(
                id=str(qid_int),
                query=txt,
                answers=[],
                answer_scores=[],
                retrieval_ids=[str(i) for i in retrieval_ids],
                subset_ids=[],
            )
            for k, v in model.model_dump().items():
                output[k].append(v)

        # Create and empty output if there are no values
        # This happends if all queries are filtered out
        # For this batch. This is necessary for the `map()` function
        # to concatenate the chunks correctly.
        if len(output) == 0:
            for k in self.output_model.model_fields:
                output[k] = []
            return output

        # check that all values are of the same length
        n_values_per_key = {k: len(v) for k, v in output.items()}
        if len(set(n_values_per_key.values())) != 1:
            raise ValueError(f"Values are not of the same length: `{n_values_per_key}`")
        return output


@fingerprint.hashregister(_FilterAndAssignRetrievalIds)
def _hash_filter_and_format_retrieval_ids(hasher: datasets.fingerprint.Hasher, obj: _FilterAndAssignRetrievalIds) -> str:
    """Register the `_FormatQueries` class to work with `datasets.map()`."""
    return hasher.hash(
        {
            "cls": obj.__class__,
            "qrels": obj.qrels._fingerprint,
            "output_model": obj.output_model,
        }
    )
