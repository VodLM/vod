import collections
import dataclasses
import functools
import json
import pathlib
import shutil
from os import PathLike
from pathlib import Path
from typing import Any, Iterable

import datasets
import fsspec
import loguru
import pydantic
from vod_configs import DatasetLoader
from vod_datasets.rosetta import models
from vod_datasets.utils import _fetch_queries_split, init_gcloud_filesystem


@dataclasses.dataclass
class RaffleSquad:
    """A Raffle-hanlded SQuAD dataset."""

    qa_splits: datasets.DatasetDict
    sections: datasets.Dataset


class SquadSectionModel(models.SectionModel):
    """A Frank section."""

    language: str

    @pydantic.field_validator("id", mode="before")
    def _validate_id(cls, value: str) -> str:
        return str(value)


class SquadQueryModel(models.QueryModel):
    """A Frank query."""

    query: str = pydantic.Field(..., alias="question")
    retrieval_ids: list[str] = pydantic.Field(..., alias="section_ids")
    category: None | str = None
    label_method_type: None | str = None
    answer_id: int
    language: str

    @pydantic.field_validator("id", mode="before")
    def _validate_id(cls, value: str) -> str:
        return str(value)

    @pydantic.field_validator("retrieval_ids", mode="before")
    def _validate_retrieval_ids(cls, value: list[str]) -> list[str]:
        return [str(v) for v in value]


def _iter_examples_from_json(
    path: PathLike | dict[str, PathLike], *, fs: fsspec.AbstractFileSystem
) -> Iterable[dict[str, Any]]:
    with fs.open(path) as f:  # type: ignore
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of dicts, got {type(data)}")

    for example in data:
        yield example


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
        yield struct_row.model_dump()


def _pase_squad_dir(local_frank_path: str, language: str, *, fs: fsspec.AbstractFileSystem) -> RaffleSquad:
    _parse_squad_dir = Path(local_frank_path, "sections.json")
    qa_splits_paths = {
        "train": Path(local_frank_path, "train_80.json"),
        "validation": Path(local_frank_path, "test.json"),
    }

    def gen_sections() -> Iterable[dict[str, Any]]:
        for section in _iter_examples_from_json(_parse_squad_dir, fs=fs):
            row = SquadSectionModel(language=language, **section)
            yield row.model_dump()

    # generate sections
    sections: datasets.Dataset = datasets.Dataset.from_generator(gen_sections)  # type: ignore

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


class RaffleSquadDatasetLoader(DatasetLoader):
    """Load the Raffle-Squad Dataset."""

    def __init__(  # noqa: PLR0913
        self,
        language: str = "en",
        cache_dir: None | str = None,
        invalidate_cache: bool = False,
        what: models.DatasetType = "queries",
    ) -> None:
        """Initialize the dataset loader."""
        self.language = language
        self.cache_dir = pathlib.Path(cache_dir or models.DATASETS_CACHE_PATH)
        self.invalidate_cache = invalidate_cache
        if what not in {"queries", "sections"}:
            raise ValueError(f"Unexpected dataset type: what=`{what}`")
        self.dataset_type = what

    def __call__(  # noqa: C901
        self,
        subset: str | None = None,
        split: None = None,
        **kws: Any,
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Load the Frank dataset."""
        if subset is not None:
            raise ValueError(f"Loader `{self.__class__.__name__}` does not support `subset` argument.")

        # define the local paths
        qa_splits_path, sections_paths = _make_local_sync_path(self.cache_dir, language=self.language)
        if self.invalidate_cache:
            loguru.logger.debug(f"Invalidating cache `{qa_splits_path}` and `{sections_paths}`")
            if qa_splits_path.exists():
                shutil.rmtree(qa_splits_path)
            if sections_paths.exists():
                shutil.rmtree(sections_paths)

        # if not downloaded, download, process and save to disk
        if not qa_splits_path.exists() or not sections_paths.exists():
            frank_split = _download_and_parse_squad(self.language)
            frank_split.qa_splits.save_to_disk(str(qa_splits_path))
            frank_split.sections.save_to_disk(str(sections_paths))

        if self.dataset_type == "sections":
            return datasets.Dataset.load_from_disk(str(sections_paths))
        if self.dataset_type == "queries":
            queries = datasets.DatasetDict.load_from_disk(str(qa_splits_path))
            return _fetch_queries_split(queries, split=split)

        raise TypeError(f"Unexpected dataset type `{self.dataset_type}`")
