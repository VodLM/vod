# noqa: I001

import os
import pathlib
import typing as typ
import uuid

import pydantic

VOD_CACHE_DIR = str(pathlib.Path(os.environ.get("VOD_CACHE_DIR", "~/.cache/vod")).expanduser())
DATASETS_CACHE_PATH = str(pathlib.Path(VOD_CACHE_DIR, "datasets"))

DatasetType = typ.Literal["queries_with_context", "queries", "sections"]


class QueryModel(pydantic.BaseModel):
    """A base query data model.

    TODO(rosetta): Remove defaults and ensure consistent types to ensure safe concatenation of datasets.
                   Having values in one dataset, but not in another leads to unexpected
                   behaviour when concatenating datasets.
                   E.g., `retrieval_ids: None | list[str]` -> `retrieval_ids: list[str]`
    """

    id: str = pydantic.Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="The unique identifier for the query.",
    )
    query: str = pydantic.Field(
        ...,
        description="The text of the question or query or instructions.",
    )
    answers: list[str] = pydantic.Field(
        default=[],
        description=(
            "A list of possible answers or completions for the query. "
            "This can be used to encoded aliases for generative tasks or answer choices for multiple choice tasks."
        ),
    )
    answer_scores: None | list[float] = pydantic.Field(
        default=None,
        description=(
            "Unnormalized scores for each answer. "
            "This can encode a multiple-choice problem or a ranking of preferences."
        ),
    )
    retrieval_ids: None | list[str] = pydantic.Field(
        default=None,
        description="A list of target section IDs `section.id` for the given query.",
    )
    retrieval_scores: None | list[float] = pydantic.Field(
        default=None,
        description=("Unnormalized scores for each retrieval ID. When not provided, assume uniform scores."),
    )
    subset_ids: None | list[str] = pydantic.Field(
        default=None,
        description="An optional ID representing a subset of data to search over.",
    )

    @pydantic.model_validator(mode="after")
    def _validate_answers_and_scores(self) -> "QueryModel":
        """Validate the answers and scores."""
        if self.answer_scores is not None and len(self.answers) != len(self.answer_scores):
            raise ValueError("The number of answers must match the number of answer scores.")
        return self

    @pydantic.model_validator(mode="after")
    def _validate_retrieval(self) -> "QueryModel":
        if (
            self.retrieval_ids is not None
            and self.retrieval_scores is not None
            and len(self.retrieval_ids) != len(self.retrieval_scores)
        ):
            raise ValueError("The number of retrieval IDs must match the number of retrieval scores.")
        return self


class SectionModel(pydantic.BaseModel):
    """A base section data model."""

    id: str = pydantic.Field(
        ...,
        description="The unique identifier for the section.",
    )
    content: str = pydantic.Field(
        ...,
        description="The main textual content of the section.",
    )
    title: None | str = pydantic.Field(
        default=None,
        description="The title of the section, if available.",
    )
    subset_id: None | str = pydantic.Field(
        default=None,
        description="An optional ID representing a subset of knowledge.",
    )


class QueryWithContextsModel(QueryModel):
    """A query with context."""

    contexts: list[str] = pydantic.Field(
        ...,
        description="The main textual content of the section.",
    )
    titles: None | list[str] = pydantic.Field(
        default=None,
        description="The title of the section, if available.",
    )

    @pydantic.model_validator(mode="after")
    def _validate_context_and_titles(self) -> "QueryWithContextsModel":
        """Validate the context and titles."""
        if self.titles is not None and len(self.titles) != len(self.contexts):
            raise ValueError("The number of titles must match the number of contexts.")
        return self
