from __future__ import annotations  # noqa: I001

import os
import pathlib
from uuid import uuid4

import datasets
import fsspec
import gcsfs
import pydantic

from vod_configs.py.utils import StrictModel

RAFFLE_PATH = str(pathlib.Path("~/.raffle").expanduser())
DATASETS_CACHE_PATH = str(pathlib.Path(RAFFLE_PATH, "datasets"))


class QueryModel(StrictModel):
    """A base query data model."""

    id: int = pydantic.Field(default_factory=uuid4, description="The unique identifier for the query.")
    query: str = pydantic.Field(
        ...,
        alias="question",
        description="The text of the question or query. This input text is used for a Language Model to process.",
    )
    answer: list = pydantic.Field(
        default=[],
        description="The generated response or answer text that a Language Model should associate to a given query. Required for generative tasks, optional for retrieval and ranking tasks.",  # noqa: E501
    )
    section_ids: list[int] = pydantic.Field(
        default=[], description="A list of IDs representing sections that contain the proper response to a query."
    )
    kb_id: int = pydantic.Field(
        default=-1,
        description="An optional ID representing a subset within the knowledge base used for searching with filtering when retrieving context for a query.",
    )
    language: str = pydantic.Field(..., description="The written language of the query, specified as a string.")

    @pydantic.validator("section_ids")
    def _validate_section_ids(cls, section_ids: list[int]) -> list[int]:
        if len(section_ids) == 0:
            raise ValueError("Section ids cannot be empty.")
        return section_ids


class SectionModel(StrictModel):
    """A base section data model."""

    content: str = pydantic.Field(..., description="The main textual content of the section.")
    title: str = pydantic.Field(default=None, description="The title of the section, if available.")
    id: int = pydantic.Field(..., description="The unique identifier for the section.")
    kb_id: int = pydantic.Field(
        default=-1,
        description="An optional ID representing the subset within the knowledge base to which the section belongs.",
    )
    language: str = pydantic.Field(
        ..., description="The written language of the section's content, specified as a string."
    )

    @pydantic.validator("title", pre=True, always=True)
    def _validate_title(cls, title: None | str) -> str:
        if title is None:
            return ""

        return title


def init_gcloud_filesystem() -> fsspec.AbstractFileSystem:
    """Initialize a GCS filesystem."""
    try:
        token = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError as exc:
        raise RuntimeError("Missing `GOOGLE_APPLICATION_CREDENTIALS` environment variables. ") from exc
    try:
        project = os.environ["GCLOUD_PROJECT_ID"]
    except KeyError as exc:
        raise RuntimeError("Missing `GCLOUD_PROJECT_ID` environment variables. ") from exc
    return gcsfs.GCSFileSystem(token=token, project=project)


def _fetch_queries_split(queries: datasets.DatasetDict, split: None | str) -> datasets.Dataset | datasets.DatasetDict:
    if split is None or split in {"all"}:
        return queries

    normalized_split = {
        "val": "validation",
    }.get(split, split)

    return queries[normalized_split]
