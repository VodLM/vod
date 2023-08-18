import typing

import pydantic


class WithContextMixin(pydantic.BaseModel):
    """A mixin for models with context."""

    section: str = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("content", "section"),
    )
    title: typing.Optional[str] = pydantic.Field(
        None,
    )


class AliasQueryModel(pydantic.BaseModel):
    """A query with an alias."""

    id: str = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("id", "query_id"),
    )
    query: str = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("query", "question"),
    )
    answer: list[str] = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("answer", "answers"),
    )


class AliasQueryWithContextModel(AliasQueryModel, WithContextMixin):
    """A query with context and aliases."""


class MultipleChoiceQueryModel(pydantic.BaseModel):
    """A multiple-choice query."""

    id: str
    query: str = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("query", "question"),
    )
    choices: list[str]
    answer: int


class AnswerDict(pydantic.BaseModel):
    """Models answer dictionaries (e.g., SQuaD)."""

    answer_start: list[int]
    text: list[str] = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("text", "value"),
    )

    @pydantic.field_validator("answer_start", mode="before")
    def _validate_answer_start(cls, answer_start: list[int]) -> list[int]:
        if not isinstance(answer_start, list):
            return [answer_start]

        return answer_start

    @pydantic.field_validator("text", mode="before")
    def _validate_text(cls, text: list[str]) -> list[str]:
        if not isinstance(text, list):
            return [text]

        return text


class AnswerDictQueryModel(pydantic.BaseModel):
    """A query with an answer dictionary."""

    id: str
    query: str = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("query", "question"),
    )
    answer: AnswerDict = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("answer", "answers"),
    )


class AnswerDictQueryWithContextModel(AnswerDictQueryModel, WithContextMixin):
    """A query with context and an answer dictionary."""
