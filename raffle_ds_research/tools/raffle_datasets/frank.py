# pylint: disable=missing-function-docstring
from __future__ import annotations

import enum
import json
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import datasets
import pydantic
import rich
from raffle_ds_storage import GoogleStorageInterface
from raffle_ds_storage.utils.static import RAFFLE_PATH
from typing_extensions import Type


class FrankSplitName(enum.Enum):
    A = "A"
    B = "B"


class HfFrankSplit(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    split: FrankSplitName
    qa_splits: datasets.DatasetDict
    sections: datasets.Dataset


class SectionModel(pydantic.BaseModel):
    content: str
    title: str
    id: int
    answer_id: int
    source_id: int
    kb_id: int = pydantic.Field(..., alias="knowledge_base_id")

    @pydantic.validator("title", pre=True, always=True)
    def _validate_title(cls, title):
        if title is None:
            return ""

        return title


class QuestionModel(pydantic.BaseModel):
    id: int
    question: str
    category: str
    label_method_type: str
    data_source: str
    answer_id: int
    section_id: Optional[int]
    kb_id: int = pydantic.Field(..., alias="knowledge_base_id")


def download_frank(
    language: str,
    version: int = 0,
    split: FrankSplitName = FrankSplitName.A,
    progress_bar: bool = True,
) -> Path:
    """Download a Frank dataset from Google Storage.

    TODO: add all sections, and not only the positive ones.
    """
    storage = GoogleStorageInterface.from_alias("datasets", progress_bar=progress_bar)
    frank_path = Path(f"datasets/frank/{language}/translated_da_frank_V{version}{split.value}/")
    storage.print_content(frank_path, recursive=True, exclude_pattern=r"kb_indexes")
    return storage.download(frank_path, exclude_pattern=r"kb_indexes")


def _read_json_data_and_create_hf(
    path: PathLike | dict[str, PathLike],
    model: Type[pydantic.BaseModel],
) -> datasets.Dataset | datasets.DatasetDict:
    if isinstance(path, dict):
        return datasets.DatasetDict({k: _read_json_data_and_create_hf(v, model) for k, v in path.items()})

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of dicts, got {type(data)}")

    output = defaultdict(list)
    column_names = None
    for row in data:
        clean_row = model(**row).dict()
        if column_names is None:
            column_names = list(clean_row.keys())

        for key in column_names:
            output[key].append(clean_row[key])

    return datasets.Dataset.from_dict(output)


def _download_and_parse_frank(
    language: str,
    split: FrankSplitName,
    version: int = 0,
) -> HfFrankSplit:
    local_frank_path = download_frank(language, version, split)
    sections_path = Path(local_frank_path, "sections.json")
    qa_splits_paths = {
        "train": Path(local_frank_path, "train_80.json"),
        "validation": Path(local_frank_path, "test.json"),
    }
    sections = _read_json_data_and_create_hf(sections_path, SectionModel)
    qa_splits = _read_json_data_and_create_hf(qa_splits_paths, QuestionModel)

    return HfFrankSplit(split=split, qa_splits=qa_splits, sections=sections)


def _make_local_sync_path(cached_dir: PathLike, language: str, split: FrankSplitName, version: int):
    base_path = Path(
        cached_dir,
        "raffle_datasets",
        "frank",
        language,
    )
    return (
        Path(base_path, f"frank_V{version}{split.value}_qa_splits.hf"),
        Path(base_path, f"frank_V{version}{split.value}_sections.hf"),
    )


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_frank(
    language: str,
    split: Union[str, FrankSplitName],
    version: int = 0,
    cache_dir: Optional[PathLike] = None,
    keep_in_memory: Optional[bool] = None,
) -> HfFrankSplit:
    """Load the Frank dataset"""
    if cache_dir is None:
        cache_dir = RAFFLE_PATH
    if isinstance(split, str):
        split = FrankSplitName(split)

    # define the local paths
    qa_splits_path, sections_paths = _make_local_sync_path(cache_dir, language=language, split=split, version=version)

    # if not downloaded, download and save to disk
    # todo: check hash before download
    #       |-> implement this feature in `raffle_ds_storage`, only download if the hash is different.
    if not qa_splits_path.exists() or not sections_paths.exists():
        frank_split = _download_and_parse_frank(language, split, version)
        frank_split.qa_splits.save_to_disk(str(qa_splits_path))
        frank_split.sections.save_to_disk(str(sections_paths))

    return HfFrankSplit(
        split=split,
        qa_splits=datasets.DatasetDict.load_from_disk(str(qa_splits_path), keep_in_memory=keep_in_memory),
        sections=datasets.Dataset.load_from_disk(str(sections_paths), keep_in_memory=keep_in_memory),
    )
