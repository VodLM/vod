from __future__ import annotations

import collections
import dataclasses
import functools
import json
import os
import pathlib
import shutil
import warnings
from typing import Iterable, Optional

import datasets
import fsspec
import loguru
import pydantic
from rich.progress import track

from .base import (
    DATASETS_CACHE_PATH,
    QueryModel,
    SectionModel,
    SilentHuggingfaceDecorator,
    init_gcloud_filesystem,
)

MS_MARCO_KB_ID = 100_000


class MsmarcoRetrievalDataset:
    """MS MARCO dataset for retrieval."""

    ...


class MsmarcoQqueryModel(QueryModel):
    """Raw MSMARCO query model."""

    data_source: str = "msmarco"
    answer_id: Optional[int] = None


class MsmarcoSectionModel(SectionModel):
    """Raw MSMARCO section model."""

    id: int = pydantic.Field(..., alias="_id")
    content: str = pydantic.Field(..., alias="text")


@dataclasses.dataclass(frozen=True)
class LocalPaths:
    """Collection of local paths for MSMARCO."""

    qa_splits: pathlib.Path
    sections: pathlib.Path


def _make_local_sync_path(cache_dir: os.PathLike, language: str = "en") -> LocalPaths:
    base_path = pathlib.Path(cache_dir, "raffle_datasets", "ms-marco", language)
    return LocalPaths(
        qa_splits=pathlib.Path(base_path, "msmarco_qa_splits.hf"),
        sections=pathlib.Path(base_path, "msmarco_sections.hf"),
    )


def _safe_decode(x: str | bytes) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _gen_queries(*, qrels: dict[int, Iterable[int]], language: str, queries_table: dict[int, str]) -> Iterable[dict]:
    meta = {"language": language, "kb_id": MS_MARCO_KB_ID}
    for qid, pids in qrels.items():
        yield MsmarcoQqueryModel(
            id=qid,
            section_ids=sorted(pids),
            text=queries_table[int(qid)],
            **meta,
        ).dict()


@SilentHuggingfaceDecorator()
def _download_and_parse_questions(language: str = "en", local_path: Optional[str] = None) -> datasets.DatasetDict:
    if language != "en":
        raise NotImplementedError("Only English MSMARCO is supported")

    if local_path is None:
        fs = init_gcloud_filesystem()
        base_path = "raffle-datasets-1/datasets/beir/msmarco/"
    else:
        fs = fsspec.filesystem("file")
        base_path = local_path

    loguru.logger.info(
        "Reading MSMARCO `{language}` questions from `{base_path}` using `{fs}`",
        language=language,
        base_path=base_path,
        fs=type(fs).__name__,
    )

    # process the queries
    queries_path = pathlib.Path(base_path, "queries.jsonl")
    qrels_paths = {
        "train": pathlib.Path(base_path, "qrels", "train.tsv"),
        "validation": pathlib.Path(base_path, "qrels", "dev.tsv"),
        "test": pathlib.Path(base_path, "qrels", "test.tsv"),
    }
    queries_table = {}
    with fs.open(str(queries_path), mode="r") as f:
        for line in track(f.readlines(), description=f"Processing `{queries_path.name}`"):
            data = json.loads(line)
            queries_table[int(data["_id"])] = data["text"]

    qa_splits = {}
    for split, qrel_path in qrels_paths.items():
        qrels: dict[int, list[int]] = collections.defaultdict(list)
        with fs.open(str(qrel_path), mode="r") as f:
            header = None
            loader = track(f.readlines(), description=f"Processing `qrels/{qrel_path.name}`")
            for _i, line in enumerate(loader):
                if header is None:
                    header = line.split()
                    if [_safe_decode(x) for x in header] != ["query-id", "corpus-id", "score"]:
                        raise ValueError(f"Unexpected header: {header}")
                    continue
                qid, pid, score = (int(x) for x in line.split())
                if score < 1:
                    # todo: check if this is correct #pylint: ignore
                    continue

                qrels[qid].append(pid)

        qa_splits[split] = datasets.Dataset.from_generator(
            functools.partial(
                _gen_queries,
                qrels=qrels,
                language=language,
                queries_table=queries_table,
            )
        )

    return datasets.DatasetDict(qa_splits)


@SilentHuggingfaceDecorator()
def _download_and_parse_sections(language: str = "en", local_path: Optional[str] = None) -> datasets.Dataset:
    if language != "en":
        raise NotImplementedError("Only English MSMARCO is supported")

    if local_path is None:
        fs = init_gcloud_filesystem()
        base_path = pathlib.Path("raffle-datasets-1/datasets/beir/msmarco/corpus.jsonl")
    else:
        fs = fsspec.filesystem("file")
        base_path = pathlib.Path(local_path, "corpus.jsonl")

    loguru.logger.info(
        "Reading MSMARCO `{language}` sections from `{base_path}` using `{fs}`",
        language=language,
        base_path=base_path,
        fs=type(fs).__name__,
    )

    def iter_sections() -> Iterable[dict]:
        meta = {"language": language, "kb_id": MS_MARCO_KB_ID}
        with fs.open(str(base_path)) as f:
            for line in track(f.readlines(), description=f"Processing `{base_path.name}`"):
                data = json.loads(line)
                yield MsmarcoSectionModel(**meta, **data).dict()

    return datasets.Dataset.from_generator(iter_sections)


def load_msmarco(
    language: str = "en",
    subset_name: Optional[str] = None,
    cache_dir: Optional[str | pathlib.Path] = None,
    keep_in_memory: Optional[bool] = None,
    only_positive_sections: bool = False,
    invalidate_cache: bool = False,
    local_source_path: Optional[str | pathlib.Path] = None,
    kb_id: Optional[int] = None,  # noqa: ARG001
) -> MsmarcoRetrievalDataset:
    """Load the MSMARCO dataset."""
    if cache_dir is None:
        cache_dir = DATASETS_CACHE_PATH
    if subset_name is not None:
        warnings.warn("subset_name is not supported for MSMARCO and will be ignored.", stacklevel=2)
    if only_positive_sections:
        raise NotImplementedError("Only positive sections is not implemented for MSMARCO")

    # define the local paths
    local_paths = _make_local_sync_path(cache_dir, language=language)
    if invalidate_cache:
        loguru.logger.debug(f"Invalidating cache `{local_paths.qa_splits}` and `{local_paths.sections}`")
        if local_paths.qa_splits.exists():
            shutil.rmtree(local_paths.qa_splits)
        if local_paths.sections.exists():
            shutil.rmtree(local_paths.sections)

    # if not downloaded, download, process and save to disk
    if not local_paths.qa_splits.exists():
        qa_splits = _download_and_parse_questions(language=language, local_path=local_source_path)
        qa_splits.save_to_disk(str(local_paths.qa_splits))
    if not local_paths.sections.exists():
        sections = _download_and_parse_sections(language=language, local_path=local_source_path)
        sections.save_to_disk(str(local_paths.sections))

    return MsmarcoRetrievalDataset(
        qa_splits=datasets.DatasetDict.load_from_disk(str(local_paths.qa_splits), keep_in_memory=keep_in_memory),
        sections=datasets.Dataset.load_from_disk(str(local_paths.sections), keep_in_memory=keep_in_memory),
    )
