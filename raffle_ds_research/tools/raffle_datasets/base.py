from __future__ import annotations

import os
import pathlib
from typing import Any, Callable, T, TypeVar

import datasets
import fsspec
import gcsfs
import pydantic

RAFFLE_PATH = str(pathlib.Path("~/.raffle").expanduser())
DATASETS_CACHE_PATH = str(pathlib.Path(RAFFLE_PATH, "datasets"))


class RetrievalDataset(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    qa_splits: datasets.DatasetDict
    sections: datasets.Dataset


def init_gcloud_filesystem() -> fsspec.AbstractFileSystem:
    try:
        token = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError as exc:
        raise RuntimeError(f"Missing `GOOGLE_APPLICATION_CREDENTIALS` environment variables. ") from exc
    try:
        project = os.environ["GCLOUD_PROJECT_ID"]
    except KeyError as exc:
        raise RuntimeError(f"Missing `GCLOUD_PROJECT_ID` environment variables. ") from exc
    return gcsfs.GCSFileSystem(token=token, project=project)


class QueryModel(pydantic.BaseModel):
    id: int
    text: str
    data_source: str
    section_ids: list[int]
    kb_id: int
    language: str

    @pydantic.validator("section_ids")
    def _validate_section_ids(cls, section_ids: list[int]) -> list[int]:
        if len(section_ids) == 0:
            raise ValueError("Section ids cannot be empty.")
        return section_ids


class SectionModel(pydantic.BaseModel):
    content: str
    title: str
    id: int
    kb_id: int
    language: str

    @pydantic.validator("title", pre=True, always=True)
    def _validate_title(cls, title: None | str) -> str:
        if title is None:
            return ""

        return title


class SilentHuggingface:
    """Silent `transformers` and `datasets` logging and progress bar."""

    def __init__(self, disable_progress_bar: bool = True, disable_logging: bool = True):
        self.disable_progress_bar = disable_progress_bar
        self.disable_logging = disable_logging

    def __enter__(self) -> None:
        if self.disable_logging:
            self._old_logging_level = datasets.utils.logging.get_verbosity()
            datasets.utils.logging.set_verbosity(datasets.logging.CRITICAL)
        if self.disable_progress_bar:
            datasets.disable_progress_bar()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.disable_logging:
            datasets.utils.logging.set_verbosity(self._old_logging_level)
        if self.disable_progress_bar:
            datasets.enable_progress_bar()


class silent_huggingface:
    def __init__(self, disable_progress_bar: bool = True, disable_logging: bool = True):
        self.disable_progress_bar = disable_progress_bar
        self.disable_logging = disable_logging

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with SilentHuggingface(self.disable_progress_bar, self.disable_logging):
                return func(*args, **kwargs)

        return wrapper
