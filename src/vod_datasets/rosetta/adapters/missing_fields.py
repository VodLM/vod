import typing
import uuid

import pydantic
from typing_extensions import Self, Type
from vod_datasets.rosetta import models
from vod_datasets.rosetta.adapters import aliases, base


class MissingFieldQueryModel(pydantic.BaseModel):
    """A query with missing fields."""

    id: str = pydantic.Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="The unique identifier for the query.",
        validation_alias=aliases.QUERY_ID_ALIASES,
    )
    query: str = pydantic.Field(
        ...,
        validation_alias=aliases.QUERY_ALIASES,
    )
    answers: None | list[str] = pydantic.Field(
        default=None,
        validation_alias=aliases.ANSWER_ALIASES,
    )
    answer_scores: None | list[float] = pydantic.Field(
        default=None,
        validation_alias=aliases.ANSWER_SCORES_ALIASES,
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


class MissingFieldQueryAdapter(base.Adapter[MissingFieldQueryModel, models.QueryModel]):
    """An adapter for multiple-choice datasets."""

    input_model = MissingFieldQueryModel
    output_model = models.QueryModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> models.QueryModel:
        """Translate a row."""
        m = cls.input_model(**row)
        # Answers
        answers = [] if m.answers is None else m.answers
        answer_scores = [1.0] * len(answers) if m.answer_scores is None else m.answer_scores
        # Retrieval
        retrieval_ids = [] if m.retrieval_ids is None else m.retrieval_ids
        retrieval_scores = [1.0] * len(retrieval_ids) if m.retrieval_scores is None else m.retrieval_scores
        # Subset
        subset_ids = [] if m.subset_ids is None else m.subset_ids
        return cls.output_model(
            id=m.id,
            query=m.query,
            answers=answers,
            answer_scores=answer_scores,
            retrieval_ids=retrieval_ids,
            retrieval_scores=retrieval_scores,
            subset_ids=subset_ids,
        )


class MissingFieldSectionModel(pydantic.BaseModel):
    """A section with missing fields."""

    id: str = pydantic.Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="The unique identifier for the query.",
        validation_alias=aliases.QUERY_ID_ALIASES,
    )
    content: str = pydantic.Field(
        ...,
        validation_alias=aliases.SECTION_ALIASES,
    )
    title: None | str = pydantic.Field(
        default=None,
        validation_alias=aliases.TITLES_ALIASES,
    )


class MissingFieldSectionAdapter(base.Adapter[MissingFieldSectionModel, models.SectionModel]):
    """An adapter for multiple-choice datasets."""

    input_model = MissingFieldSectionModel
    output_model = models.SectionModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> models.SectionModel:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(
            id=m.id,
            content=m.content,
            title=m.title,
        )
