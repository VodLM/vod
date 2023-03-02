from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Literal, List, Optional, Tuple

import datasets
import rich
from datasets import Dataset as HfDataset
from datasets import DatasetDict as HfDatasetDict
from loguru import logger
from pydantic import BaseModel, ValidationError


def model_path(language: str) -> Path:
    return Path(f"production_pretrain/ranking/{language}/frank")


def dataset_path(language: str):
    return Path(f"datasets/frank/{language}")


def local_dataset_path(language: str):
    return Path("~/.raffle").expanduser() / dataset_path(language)


def squad_path(language: str):
    return Path(f"datasets/squad/{language}/squad_{language}")


def local_squad_path(language: str):
    return Path("~/.raffle").expanduser() / squad_path(language)


def add_sections(ds: Dataset) -> Dataset:
    if not ds.sections:
        for a in ds.answers:
            ds.sections.append(
                Section(
                    a.text,
                    a.title,
                    a.answer_id,
                    a.answer_id,
                    a.version,
                    None,
                    a.source_id,
                    None,
                )
            )
    return ds


FrankSplitName = Literal["A", "B"]
FrankTypeName = Literal["qa_splits", "sections"]


class FrankDsetInfo(BaseModel):
    """Models the split, version, remote and local paths for a Frank dataset."""

    version: int
    split: FrankSplitName
    path: str
    local_path: Path


def download_latest_frank_datasets(
    language: str,
) -> Dict[FrankSplitName, FrankDsetInfo]:
    gc_dsi = GoogleCloudStorageInterface("dataset")
    dataset_folder = dataset_path(language)
    datasets = gc_dsi.list_folders(str(dataset_folder), recursive=False)
    datasets = [f"{dataset_folder}/{d}" for d in datasets]
    rich.print(f"Datasets: {datasets}")

    if len(datasets) == 0:
        raise ValueError(f"No datasets found for {dataset_folder}")

    def _parse_dset_info(ds_remote_path: str) -> None | FrankDsetInfo:
        """Parse the version number and split from a dataset path"""
        pattern = re.compile(r"_V(\d+?)([AB])")
        match = re.search(pattern, ds_remote_path)
        if match is None:
            return None

        local_path = Path(RAFFLE_PATH, ds_remote_path)
        return FrankDsetInfo(
            version=int(match.group(1)),
            split=match.group(2),
            path=ds_remote_path,
            local_path=local_path,
        )

    # Fetch the paths for the latest version of each split
    dsets_by_split = defaultdict(list)
    for ds in datasets:
        info = _parse_dset_info(ds)
        if info is not None:
            dsets_by_split[info.split].append(info)

    dsets_by_split = {split: max(infos, key=lambda x: x.version) for split, infos in dsets_by_split.items()}

    # Download the latest version of each split
    for split, info in dsets_by_split.items():
        logger.debug(f"Downloading split {split} from `{info.path}` to {info.local_path}")
        gc_dsi.download_folder(info.path, RAFFLE_PATH)
        if not info.local_path.exists():
            raise FileNotFoundError(
                f"Could not find {info.local_path}. " f"Probably something went wrong when downloading the files."
            )

    return dsets_by_split


def get_latest_datasets(
    language: str,
) -> Dict[FrankSplitName, Tuple[FrankDsetInfo, Dataset]]:
    local_paths = download_latest_frank_datasets(language)

    dsets = {}
    for split, info in local_paths.items():
        dset = Dataset.from_directory(local_dataset_path(language) / info.local_path.name)
        dsets[split] = (info, dset)

    return dsets


class DtypesQaModel(BaseModel):
    """Models a QA example"""

    id: int
    question: str
    category: str
    label_method_type: str
    data_source: str
    answer_id: int
    section_id: Optional[int]
    knowledge_base_id: int


QaSplits = dict  # dict[str, List[QuestionAnswer]]
Sections = list  # List[Section]


def dtypes_qa_to_hf(qa_splits: QaSplits) -> HfDatasetDict:
    """Converts a QA split to a HuggingFace Dataset"""
    qa_splits = rename_qa_splits(qa_splits)
    dset_splits = {}
    for split, qa_split in qa_splits.items():
        qa_split_dict = defaultdict(list)
        for qa in qa_split:
            qa = DtypesQaModel(**qa.to_dict())
            for key, value in qa.dict().items():
                qa_split_dict[key].append(value)

        hf_dset = datasets.Dataset.from_dict(qa_split_dict)
        dset_splits[split] = hf_dset

    return HfDatasetDict(dset_splits)


def rename_qa_splits(qa_splits: QaSplits) -> QaSplits:
    mapping = {
        "train_80": "train",
        "test": "validation",
    }

    new_splits = {}
    for old, new in mapping.items():
        new_splits[new] = qa_splits[old]

    return new_splits


class DtypesSectionModel(BaseModel):
    """Models a Section"""

    id: int
    content: str
    title: Optional[str]
    answer_id: int
    knowledge_base_id: int
    parent_section_id: Optional[int]
    source_id: int
    answer_version: int


def dtypes_sections_to_hf(sections: Sections) -> HfDataset:
    """Converts a list of Sections to a HuggingFace Dataset"""
    hf_sections = defaultdict(list)
    for section in sections:
        sec_dict = section.to_dict()
        try:
            section = DtypesSectionModel(**sec_dict)
        except ValidationError as e:
            rich.print(sec_dict)
            raise e
        for key, value in section.dict().items():
            hf_sections[key].append(value)

    hf_dset = datasets.Dataset.from_dict(hf_sections)

    return hf_dset


class HfFrankSplit(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    qa_splits: HfDatasetDict
    sections: HfDataset


class HfFrank(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    A: HfFrankSplit
    B: HfFrankSplit


def load_hf_frank(language: str):
    dset_paths = {
        "A": {
            "qa_splits": local_dataset_path(language) / "translated_da_frank_V0A_qa_splits.hf",
            "sections": local_dataset_path(language) / "translated_da_frank_V0A_sections.hf",
        },
        "B": {
            "qa_splits": local_dataset_path(language) / "translated_da_frank_V0B_qa_splits.hf",
            "sections": local_dataset_path(language) / "translated_da_frank_V0B_sections.hf",
        },
    }

    def _load_hf_frank() -> HfFrank:
        hf_dsets = defaultdict(dict)
        for split, paths in dset_paths.items():
            for name, path in paths.items():
                logger.debug(f"Loading {name} ({split}) from {path}...")
                hf_dsets[split][name] = datasets.load_from_disk(path)

        return HfFrank(**hf_dsets)

    def _download_convert_save_frank(language):
        dsets = get_latest_datasets(language)
        infos = {split: dset[0] for split, dset in dsets.items()}
        dsets = {split: dset[1] for split, dset in dsets.items()}
        # convert the QA splits
        qa_splits = {split: dset.qa_splits for split, dset in dsets.items()}
        hf_qa_dsets = {split: dtypes_qa_to_hf(qa) for split, qa in qa_splits.items()}
        for split, dset in hf_qa_dsets.items():
            info = infos[split]
            save_path = info.local_path.parent / f"{info.local_path.name}_qa_splits.hf"
            logger.debug(f"Saving QA splits ({split}) to {save_path}...")
            dset.save_to_disk(save_path)
        # convert the sections
        sections = {split: dset.sections for split, dset in dsets.items()}
        hf_sections = {split: dtypes_sections_to_hf(sections) for split, sections in sections.items()}
        for split, dset in hf_sections.items():
            info = infos[split]
            save_path = info.local_path.parent / f"{info.local_path.name}_sections.hf"
            logger.debug(f"Saving sections ({split}) to {save_path}...")
            dset.save_to_disk(save_path)

    try:
        hf_dsets = _load_hf_frank()

    except FileNotFoundError:
        _download_convert_save_frank(language)
        hf_dsets = _load_hf_frank()

    return hf_dsets
