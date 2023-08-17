from __future__ import annotations

import collections
import dataclasses
import functools
import json
import pathlib
import re
import shutil
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Optional

import datasets
import fsspec
import loguru
import pydantic
from rich.progress import track

from src import vod_configs

from .base import (
    DATASETS_CACHE_PATH,
    QueryModel,
    SectionModel,
    SilentHuggingfaceDecorator,
    _fetch_queries_split,
    init_gcloud_filesystem,
)

FrankDsetNamePattern = re.compile(r"^(?P<name>[frank]+)(_(?P<frank_split>[A-Za-z_]))$")


class FrankSectionModel(SectionModel):
    """A Frank section."""

    section: str = pydantic.Field(..., alias="content")
    kb_id: int = pydantic.Field(..., alias="knowledge_base_id")
    answer_id: int


class FrankQueryModel(QueryModel):
    """A Frank query."""

    query: str = pydantic.Field(..., alias="question")
    category: Optional[str] = None
    label_method_type: Optional[str] = None
    answer_id: int
    kb_id: int = pydantic.Field(..., alias="knowledge_base_id")


@dataclasses.dataclass
class HfFrankPart:
    """A part of the Frank dataset (A or B)."""

    qa_splits: datasets.DatasetDict
    sections: datasets.Dataset
    split: str

    def __add__(self, other: "HfFrankPart") -> "HfFrankPart":
        """Merge two parts of the Frank dataset."""
        if not isinstance(other, HfFrankPart):
            raise TypeError(f"Expected HfFrankPart, got {type(other)}")
        if not self.split == other.split:
            raise ValueError(f"Expected split {self.split}, got {other.split}")

        if not set(self.qa_splits.keys()) == set(other.qa_splits.keys()):
            raise ValueError(f"Expected qa_splits {set(self.qa_splits.keys())}, got {set(other.qa_splits.keys())}")

        qa_splits = datasets.DatasetDict(
            {k: datasets.concatenate_datasets([self.qa_splits[k], other.qa_splits[k]]) for k in self.qa_splits}
        )

        sections = datasets.concatenate_datasets([self.sections, other.sections])

        return HfFrankPart(split=self.split, qa_splits=qa_splits, sections=sections)


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
def _download_and_parse_frank(
    language: str,
    split: str,
    version: int = 0,
    only_positive_sections: bool = False,
) -> HfFrankPart:
    fs = init_gcloud_filesystem()
    if only_positive_sections:
        if language == "da":
            path = f"raffle-datasets-1/datasets/frank/{language}/da_frank_V{version}{split}/"
        else:
            path = f"raffle-datasets-1/datasets/frank/{language}/translated_da_frank_V{version}{split}/"
        loguru.logger.debug(f"Reading Frank from {path} using {fs}")
        if not fs.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist on storage {fs}.")
        return _pase_frank_dir(path, split, language=language, fs=fs)  # type: ignore

    if language == "da":
        path = f"raffle-datasets-1/datasets/frank/{language}/da_frank_V{version}{split}/kb_indexes/"
    else:
        path = f"raffle-datasets-1/datasets/frank/{language}/translated_da_frank_V{version}{split}/kb_indexes/"
    loguru.logger.debug(f"Reading Frank from {path} using {fs}")
    if not fs.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist on storage {fs}.")
    full_frank_split = None
    for part_path in track(fs.ls(path), description=f"Processing Frank {split}"):
        if not fs.exists(Path(part_path, "sections.json")):
            loguru.logger.debug(f"Skipping {part_path} (no sections.json)")
            continue
        frank_part = _pase_frank_dir(part_path, split, language=language, fs=fs)  # type: ignore
        if full_frank_split is None:
            full_frank_split = frank_part
        else:
            full_frank_split += frank_part
    if full_frank_split is None:
        raise ValueError(f"Frank {split} is empty.")
    return full_frank_split


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
        struct_row = FrankQueryModel(language=language, **row)
        yield struct_row.dict()


def _pase_frank_dir(local_frank_path: str, split: str, language: str, *, fs: fsspec.AbstractFileSystem) -> HfFrankPart:
    sections_path = Path(local_frank_path, "sections.json")
    qa_splits_paths = {
        "train": Path(local_frank_path, "train_80.json"),
        "validation": Path(local_frank_path, "test.json"),
    }

    def gen_sections() -> Iterable[dict[str, Any]]:
        for section in _iter_examples_from_json(sections_path, fs=fs):
            struct_row = FrankSectionModel(language=language, **section)
            yield struct_row.dict()

    # generate sections
    sections: datasets.Dataset = datasets.Dataset.from_generator(gen_sections)  # type: ignore

    # build `answer2section`
    answer2section: dict[int, set[int]] = collections.defaultdict(set)
    for section in _iter_examples_from_json(sections_path, fs=fs):
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

    return HfFrankPart(split=split, qa_splits=qa_splits, sections=sections)


def _make_local_sync_path(
    cache_dir: str | pathlib.Path,
    language: str,
    split: str,
    version: int,
    only_positive_sections: bool = False,
) -> tuple[pathlib.Path, pathlib.Path]:
    base_path = pathlib.Path(
        cache_dir, "raffle_datasets", "frank", language, "only-positives" if only_positive_sections else "full"
    )
    return (
        pathlib.Path(base_path, f"frank_V{version}{split}_qa_splits.hf"),
        pathlib.Path(base_path, f"frank_V{version}{split}_sections.hf"),
    )


def load_frank(
    config: vod_configs.BaseDatasetConfig,
    *,
    only_positive_sections: bool = False,
    version: int = 0,
) -> datasets.Dataset | datasets.DatasetDict:
    """Load the Frank dataset."""
    cache_dir = pathlib.Path(config.options.cache_dir or DATASETS_CACHE_PATH)

    # Parse the frank split
    parsed_named = FrankDsetNamePattern.match(config.name)
    if parsed_named is None or "frank_split" not in parsed_named.groupdict():
        raise ValueError(f"Failed to parse Frank name `{config.name}` using pattern `{FrankDsetNamePattern.pattern}`")
    frank_split = parsed_named.groupdict()["frank_split"].upper()

    # Check the language (split)
    if config.subset is None:
        raise ValueError(f"Split must be configured. Config={config}")

    # Define the local paths
    qa_splits_path, sections_paths = _make_local_sync_path(
        cache_dir,
        language=config.subset,
        split=frank_split,
        version=version,
        only_positive_sections=only_positive_sections,
    )
    if config.options.invalidate_cache:
        loguru.logger.debug(f"Invalidating cache `{qa_splits_path}` and `{sections_paths}`")
        if qa_splits_path.exists():
            shutil.rmtree(qa_splits_path)
        if sections_paths.exists():
            shutil.rmtree(sections_paths)

    # If not downloaded, download, process and save to disk
    if not qa_splits_path.exists() or not sections_paths.exists():
        frank_dset = _download_and_parse_frank(config.subset, frank_split, version, only_positive_sections)
        frank_dset.qa_splits.save_to_disk(str(qa_splits_path))
        frank_dset.sections.save_to_disk(str(sections_paths))

    if isinstance(config, vod_configs.SectionsDatasetConfig):
        return datasets.Dataset.load_from_disk(str(sections_paths))

    if isinstance(config, vod_configs.QueriesDatasetConfig):
        queries = datasets.DatasetDict.load_from_disk(str(qa_splits_path))
        return _fetch_queries_split(queries, split=config.split)

    raise TypeError(f"Unexpected config type {type(config)}")


def _filter_by_kbid(row: dict, idx: Optional[int] = None, *, kb_id: int) -> bool:  # noqa: ARG001
    return int(kb_id) == int(row["kb_id"])
