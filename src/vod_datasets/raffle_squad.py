from __future__ import annotations

import collections
import functools
import json
import pathlib
import shutil
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import datasets
import fsspec
import loguru
import pydantic
from raffle_ds_research.tools.raffle_datasets.base import (
    DATASETS_CACHE_PATH,
    QueryModel,
    RetrievalDataset,
    SectionModel,
    SilentHuggingfaceDecorator,
    init_gcloud_filesystem,
)

RAFFLE_SQUAD_KB_ID = 200_001


class RaffleSquad(RetrievalDataset):
    """A Raffle-hanlded SQuAD dataset."""


class SquadSectionModel(SectionModel):
    """A Frank section."""

    kb_id: int = RAFFLE_SQUAD_KB_ID
    answer_id: int


class SquadQueryModel(QueryModel):
    """A Frank query."""

    text: str = pydantic.Field(..., alias="question")
    category: Optional[str] = None
    label_method_type: Optional[str] = None
    answer_id: int
    kb_id: int = RAFFLE_SQUAD_KB_ID


def _iter_examples_from_json(
    path: PathLike | dict[str, PathLike], *, fs: fsspec.AbstractFileSystem
) -> Iterable[dict[str, Any]]:
    with fs.open(path) as f:  # type: ignore
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of dicts, got {type(data)}")

    for example in data:
        yield example


@SilentHuggingfaceDecorator()
def _download_and_parse_squad(language: str) -> RaffleSquad:
    fs = init_gcloud_filesystem()

    path = f"raffle-datasets-1/datasets/squad/{language}/squad_{language}/"
    loguru.logger.debug(f"Reading Raffle SQuAD from {path} using {fs}")
    if not fs.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist on storage {fs}.")

    return _pase_squad_dir(path, language=language, fs=fs)


def _gen_qa_split(
    qa_split_path: PathLike | dict[str, PathLike],
    fs: fsspec.AbstractFileSystem,
    language: str,
    answer2section: dict[int, set[int]],
) -> Iterable[dict[str, Any]]:
    for row in _iter_examples_from_json(qa_split_path, fs=fs):
        answer_id = int(row["answer_id"])
        section_id = row["section_id"]
        if section_id is None:
            section_ids = answer2section[answer_id]
            section_ids = sorted(section_ids)
            row["section_ids"] = section_ids
        else:
            section_id = int(section_id)
            row["section_ids"] = [section_id]
        struct_row = SquadQueryModel(language=language, **row)
        yield struct_row.dict()


def _pase_squad_dir(local_frank_path: str, language: str, *, fs: fsspec.AbstractFileSystem) -> RaffleSquad:
    _parse_squad_dir = Path(local_frank_path, "sections.json")
    qa_splits_paths = {
        "train": Path(local_frank_path, "train_80.json"),
        "validation": Path(local_frank_path, "test.json"),
    }

    def gen_sections() -> Iterable[dict[str, Any]]:
        for section in _iter_examples_from_json(_parse_squad_dir, fs=fs):
            row = SquadSectionModel(language=language, **section)
            yield row.dict()

    # generate sections
    sections = datasets.Dataset.from_generator(gen_sections)

    # build `answer2section`
    answer2section: dict[int, set[int]] = collections.defaultdict(set)
    for section in _iter_examples_from_json(_parse_squad_dir, fs=fs):
        section_id = int(section["answer_id"])
        answer2section[section_id].add(int(section["id"]))

    # generate QA splits
    qa_splits = {}
    for split_name, qa_split_path in qa_splits_paths.items():
        qa_splits[split_name] = datasets.Dataset.from_generator(
            functools.partial(
                _gen_qa_split,
                qa_split_path=qa_split_path,
                fs=fs,
                language=language,
                answer2section=answer2section,
            )
        )
    qa_splits = datasets.DatasetDict(qa_splits)

    return RaffleSquad(qa_splits=qa_splits, sections=sections)


def _make_local_sync_path(cache_dir: str | pathlib.Path, language: str) -> tuple[pathlib.Path, pathlib.Path]:
    base_path = pathlib.Path(cache_dir, "raffle_datasets", "squad", language, "full")
    return (
        pathlib.Path(base_path, "raffle_squad_qa_splits.hf"),
        pathlib.Path(base_path, "raffle_squad_sections.hf"),
    )


def load_raffle_squad(
    language: str = "en",
    cache_dir: Optional[Union[str, pathlib.Path]] = None,
    keep_in_memory: Optional[bool] = None,
    invalidate_cache: bool = False,
    subset_name: Optional[str] = None,  # noqa: ARG001
    only_positive_sections: Optional[bool] = None,  # noqa: ARG001
    kb_id: Optional[int] = None,  # noqa: ARG001
) -> RaffleSquad:
    """Load the Frank dataset."""
    if cache_dir is None:
        cache_dir = pathlib.Path(DATASETS_CACHE_PATH)

    # define the local paths
    qa_splits_path, sections_paths = _make_local_sync_path(cache_dir, language=language)
    if invalidate_cache:
        loguru.logger.debug(f"Invalidating cache `{qa_splits_path}` and `{sections_paths}`")
        if qa_splits_path.exists():
            shutil.rmtree(qa_splits_path)
        if sections_paths.exists():
            shutil.rmtree(sections_paths)

    # if not downloaded, download, process and save to disk
    if not qa_splits_path.exists() or not sections_paths.exists():
        frank_split = _download_and_parse_squad(language)
        frank_split.qa_splits.save_to_disk(str(qa_splits_path))
        frank_split.sections.save_to_disk(str(sections_paths))

    return RaffleSquad(
        qa_splits=datasets.DatasetDict.load_from_disk(str(qa_splits_path), keep_in_memory=keep_in_memory),
        sections=datasets.Dataset.load_from_disk(str(sections_paths), keep_in_memory=keep_in_memory),
    )
