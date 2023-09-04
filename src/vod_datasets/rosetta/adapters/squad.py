import typing
import uuid

import pydantic
from typing_extensions import Self, Type
from vod_datasets.rosetta import models
from vod_datasets.rosetta.adapters import aliases, base, contextual


class AnswerDict(pydantic.BaseModel):
    """Models answer dictionaries (e.g., SQuaD)."""

    text: list[str]


class SquadQueryModel(pydantic.BaseModel):
    """A query with an answer dictionary."""

    id: str = pydantic.Field(
        default_factory=lambda: uuid.uuid4().hex,
        validation_alias=aliases.QUERY_ID_ALIASES,
    )
    query: str = pydantic.Field(
        ...,
        validation_alias=aliases.QUERY_ALIASES,
    )
    answer: AnswerDict = pydantic.Field(
        ...,
        validation_alias=aliases.ANSWER_ALIASES,
    )


class SquadQueryWithContextsModel(SquadQueryModel, contextual.WithContexstMixin):
    """A query with context and an answer dictionary."""


class SquadQueryAdapter(base.Adapter[SquadQueryModel, models.QueryModel]):
    """Handle Squad-like answer dicts."""

    input_model = SquadQueryModel
    output_model = models.QueryModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> models.QueryModel:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(
            id=m.id,
            query=m.query,
            answers=m.answer.text,
            answer_scores=[1.0] * len(m.answer.text),
        )


class SquadQueryWithContextsAdapter(base.Adapter[SquadQueryWithContextsModel, models.QueryWithContextsModel]):
    """Handle Squad-like answer dicts with context dicts (e.g., TriviaQA)."""

    input_model = SquadQueryWithContextsModel
    output_model = models.QueryWithContextsModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> models.QueryWithContextsModel:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(
            id=m.id,
            query=m.query,
            answers=m.answer.text,
            answer_scores=[1.0] * len(m.answer.text),
            contexts=m.contexts,
            titles=m.titles,
        )
