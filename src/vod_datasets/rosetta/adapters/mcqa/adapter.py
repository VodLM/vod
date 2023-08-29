import typing
import uuid

import pydantic
from typing_extensions import Self, Type
from vod_datasets.rosetta import models
from vod_datasets.rosetta.adapters import adapter, aliases, contextual

ANSWER_CHOICES_LETTERS = typing.Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def _answer_to_int(x: int | ANSWER_CHOICES_LETTERS) -> int:
    """Convert an answer to an integer."""
    if isinstance(x, int):
        return x
    values = typing.get_args(ANSWER_CHOICES_LETTERS)
    if x not in values:
        raise ValueError(f"Invalid answer: `{x}`")
    return values.index(x)


class MultipleChoiceQueryModel(pydantic.BaseModel):
    """A multiple-choice query."""

    id: str = pydantic.Field(
        default=uuid.uuid4().hex,
        description="The unique identifier for the query.",
        validation_alias=aliases.QUERY_ID_ALIASES,
    )
    query: str = pydantic.Field(
        ...,
        validation_alias=aliases.QUERY_ALIASES,
    )
    choices: list[str] = pydantic.Field(
        ...,
        validation_alias=aliases.CHOICES_ALIASES,
    )
    answer: typing.Union[int, ANSWER_CHOICES_LETTERS] = pydantic.Field(
        ...,
        validation_alias=aliases.ANSWER_CHOICE_IDX_ALIASES,
    )


class MultipleChoiceQueryAdapter(adapter.Adapter[MultipleChoiceQueryModel, models.QueryModel]):
    """An adapter for multiple-choice datasets."""

    input_model = MultipleChoiceQueryModel
    output_model = models.QueryModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> models.QueryModel:
        """Translate a row."""
        m = cls.input_model(**row)
        a_idx = _answer_to_int(m.answer)
        return cls.output_model(
            id=m.id,
            query=m.query,
            choices=m.choices,
            answer=[m.choices[a_idx]],
        )


class MultipleChoiceQueryWithContextsModel(MultipleChoiceQueryModel, contextual.WithContexstMixin):
    """A query with context and an answer dictionary."""


class MultipleChoiceQueryWithContextAdapter(
    adapter.Adapter[MultipleChoiceQueryWithContextsModel, models.QueryWithContextsModel]
):
    """An adapter for contextual multiple-choice queries."""

    input_model = MultipleChoiceQueryWithContextsModel
    output_model = models.QueryWithContextsModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> models.QueryWithContextsModel:
        """Translate a row."""
        m = cls.input_model(**row)
        a_idx = _answer_to_int(m.answer)
        return cls.output_model(
            id=m.id,
            query=m.query,
            choices=m.choices,
            answer=[m.choices[a_idx]],
            contexts=m.contexts,
            titles=m.titles,
        )
