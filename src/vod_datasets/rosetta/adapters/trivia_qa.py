import typing

import pydantic
from typing_extensions import Self, Type
from vod_datasets.rosetta import models
from vod_datasets.rosetta.adapters import base


class AnswerDict(pydantic.BaseModel):
    """Models an answer dict for TriviaQA."""

    aliases: list[str]


class TriviaQaQueryModel(pydantic.BaseModel):
    """TriviaQA query."""

    question_id: str
    question: str
    answer: AnswerDict


class EntityPages(pydantic.BaseModel):
    """A mixin for models with context."""

    title: list[str]
    wiki_context: list[str]


class TriviaQaQueryWithContextsModel(TriviaQaQueryModel):
    """Models a section with a contexts dictionary."""

    entity_pages: EntityPages


class TriviaQaQueryAdapter(base.Adapter[TriviaQaQueryModel, models.QueryModel]):
    """Handle Squad-like answer dicts with context dicts (e.g., TriviaQA)."""

    input_model = TriviaQaQueryModel
    output_model = models.QueryModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> models.QueryModel:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(
            id=m.question_id,
            query=m.question,
            answers=m.answer.aliases,
            answer_scores=[1.0] * len(m.answer.aliases),
        )


class TriviaQaQueryWithContextsAdapter(base.Adapter[TriviaQaQueryWithContextsModel, models.QueryWithContextsModel]):
    """Handle Squad-like answer dicts with context dicts (e.g., TriviaQA)."""

    input_model = TriviaQaQueryWithContextsModel
    output_model = models.QueryWithContextsModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> models.QueryWithContextsModel:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(
            id=m.question_id,
            query=m.question,
            answers=m.answer.aliases,
            answer_scores=[1.0] * len(m.answer.aliases),
            contexts=m.entity_pages.wiki_context,
            titles=m.entity_pages.title,
        )
