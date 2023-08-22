from __future__ import annotations  # noqa: I001

import os
import pathlib
import typing
import uuid

import pydantic

VOD_CACHE_DIR = str(pathlib.Path(os.environ.get("VOD_CACHE_DIR", "~/.cache/vod")).expanduser())
DATASETS_CACHE_PATH = str(pathlib.Path(VOD_CACHE_DIR, "datasets"))

ModelType = typing.Literal["query_with_context", "query", "section"]


class QueryModel(pydantic.BaseModel):
    """A base query data model."""

    id: typing.Union[str, int] = pydantic.Field(
        default=uuid.uuid4().hex,
        description="The unique identifier for the query.",
    )
    query: str = pydantic.Field(
        ...,
        description="The text of the question or query. This input text is used for a Language Model to process.",
    )
    choices: list[str] = pydantic.Field(
        default=[],
        description=(
            "A list of strings representing the possible answers to the query. "
            "Required for retrieval and ranking tasks, optional for generative tasks."
        ),
    )
    answer: list[str] = pydantic.Field(
        default=[],
        description=(
            "The generated response or answer text that a Language Model should associate to a given query. "
            "Required for generative tasks, optional for retrieval and ranking tasks."
        ),
    )
    section_ids: list[int] = pydantic.Field(
        default=[],
        description="A list of IDs representing sections that contain the proper response to a query.",
    )
    subset_id: int = pydantic.Field(
        default=-1,
        description="An optional ID representing a subset of data",
    )
    language: str = pydantic.Field(
        default="en",
        description="The written language of the query, specified as a string.",
    )


class SectionModel(pydantic.BaseModel):
    """A base section data model."""

    id: typing.Union[str, int] = pydantic.Field(
        default=uuid.uuid4().hex,
        description="The unique identifier for the section.",
    )
    section: str = pydantic.Field(
        ...,
        description="The main textual content of the section.",
    )
    title: typing.Optional[str] = pydantic.Field(
        default=None,
        description="The title of the section, if available.",
    )
    subset_id: int = pydantic.Field(
        default=-1,
        description="An optional ID representing a target subset of sections.",
    )
    language: str = pydantic.Field(
        default="en",
        description="The written language of the section's content, specified as a string.",
    )


class QueryWithContextModel(QueryModel, SectionModel):
    """A query with context."""
