from __future__ import annotations  # noqa: I001

import os
import pathlib
import typing
import uuid

import pydantic

VOD_CACHE_DIR = str(pathlib.Path(os.environ.get("VOD_CACHE_DIR", "~/.cache/vod")).expanduser())
DATASETS_CACHE_PATH = str(pathlib.Path(VOD_CACHE_DIR, "datasets"))

DatasetType = typing.Literal["queries_with_context", "queries", "sections"]


class QueryModel(pydantic.BaseModel):
    """A base query data model."""

    id: str = pydantic.Field(
        default=uuid.uuid4().hex,
        description="The unique identifier for the query.",
    )
    query: str = pydantic.Field(
        ...,
        description="The text of the question or query. This input text is used for a Language Model to process.",
    )
    answer: list[str] = pydantic.Field(
        default=[],
        description=(
            "The generated response or answer text that a Language Model should associate to a given query. "
            "Required for generative tasks, optional for retrieval and ranking tasks."
        ),
    )
    choices: list[str] = pydantic.Field(
        default=[],
        description=(
            "A list of strings representing the possible answers to the query. "
            "Required for retrieval and ranking tasks, optional for generative tasks."
        ),
    )
    section_ids: list[str] = pydantic.Field(
        default=[],
        description="A list of labels `section.uid`.",
    )
    subset_ids: list[str] = pydantic.Field(
        default=[],
        description="An optional ID representing a subset of data",
    )
    language: str = pydantic.Field(
        default="en",
        description="The written language of the query, specified as a string.",
    )


class SectionModel(pydantic.BaseModel):
    """A base section data model."""

    id: str = pydantic.Field(
        default=uuid.uuid4().hex,
        description="The unique identifier for the section.",
    )
    content: str = pydantic.Field(
        ...,
        description="The main textual content of the section.",
    )
    title: typing.Optional[str] = pydantic.Field(
        default=None,
        description="The title of the section, if available.",
    )
    subset_id: typing.Optional[str] = pydantic.Field(
        default=None,
        description="An optional ID representing a target subset of sections.",
    )
    language: str = pydantic.Field(
        default="en",
        description="The written language of the section's content, specified as a string.",
    )


class QueryWithContextsModel(QueryModel):
    """A query with context."""

    contexts: list[str] = pydantic.Field(
        ...,
        description="The main textual content of the section.",
    )
    titles: typing.Optional[list[str]] = pydantic.Field(
        default=None,
        description="The title of the section, if available.",
    )

    @pydantic.model_validator(mode="after")
    def _validate_context_and_titles(self) -> "QueryWithContextsModel":
        """Validate the context and titles."""
        if self.titles is not None and len(self.titles) != len(self.contexts):
            raise ValueError("The number of titles must match the number of contexts.")
        return self
