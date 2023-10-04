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
    answers: list[str] = pydantic.Field(
        default=[],
        validation_alias=aliases.ANSWER_ALIASES,
    )
    answer_scores: list[float] = pydantic.Field(
        default=None,
        validation_alias=aliases.ANSWER_SCORES_ALIASES,
    )


class MissingFieldQueryAdapter(base.Adapter[MissingFieldQueryModel, models.QueryModel]):
    """An adapter for multiple-choice datasets."""

    input_model = MissingFieldQueryModel
    output_model = models.QueryModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> models.QueryModel:
        """Translate a row."""
        m = cls.input_model(**row)
        scores = [1.0] * len(m.answers) if m.answer_scores is None else m.answer_scores
        return cls.output_model(
            id=m.id,
            query=m.query,
            answers=m.answers,
            answer_scores=scores,
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
