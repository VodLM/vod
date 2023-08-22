import typing
import uuid

import pydantic
from vod_datasets.models import QueryModel, SectionModel

QUERY_ID_ALIASES = pydantic.AliasChoices("id", "question_id", "query_id")
SECTION_ID_ALIASES = pydantic.AliasChoices("id", "section_id", "context_id", "passage_id")
QUERY_ALIASES = pydantic.AliasChoices("query", "question")
SECTION_ALIASES = pydantic.AliasChoices("passage", "context", "section", "content", "text")
ANSWER_ALIASES = pydantic.AliasChoices("answer", "answers", "response")
ANSWER_KEY_ALIASES = pydantic.AliasChoices("value", "text")
CHOICES_ALIASES = pydantic.AliasChoices("choices", "options", "candidates")

class WithContextMixin(pydantic.BaseModel):
    """A mixin for models with context."""

    section: str = pydantic.Field(
        ...,
        validation_alias=SECTION_ALIASES,
    )
    title: typing.Optional[str] = pydantic.Field(
        None,
    )


class AliasQueryModel(QueryModel):
    """A query with an alias."""

    query: str = pydantic.Field(
        ...,
        validation_alias=QUERY_ALIASES,
    )
    answer: list[str] = pydantic.Field(
        ...,
        validation_alias=ANSWER_ALIASES,
    )


class AliasSectionModel(SectionModel):
    """A query with an alias."""
    section: str = pydantic.Field(
        ...,
        validation_alias=SECTION_ALIASES,
    )
    title: list[str] = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("title"),
    )

class AliasQueryWithContextModel(AliasQueryModel, WithContextMixin):
    """A query with context and aliases."""


class MultipleChoiceQueryModel(pydantic.BaseModel):
    """A multiple-choice query."""

    id: str = pydantic.Field(
        default=uuid.uuid4().hex,
        description="The unique identifier for the query.",
        validation_alias=QUERY_ID_ALIASES,
    )
    query: str = pydantic.Field(
        ...,
        validation_alias=QUERY_ALIASES,
    )
    choices: list[str] = pydantic.Field(
        ...,
        validation_alias=CHOICES_ALIASES,
    )
    answer: int


class AnswerDict(pydantic.BaseModel):
    """Models answer dictionaries (e.g., SQuaD)."""
    
    text: str | list[str] = pydantic.Field(
        None,
        validation_alias=ANSWER_KEY_ALIASES,
    )

    @pydantic.field_validator("text")
    def validate_answer_value(cls, value: str | list[str]) -> list[str]:
        """Validate the dataset name or path."""
        if isinstance(value, str):
            return [value]
        return value


class AnswerDictQueryModel(pydantic.BaseModel):
    """A query with an answer dictionary."""

    id: str = pydantic.Field(
        default=uuid.uuid4().hex,
        validation_alias=QUERY_ID_ALIASES,
    )
    query: str = pydantic.Field(
        ...,
        validation_alias=QUERY_ALIASES,
    )
    answer: AnswerDict = pydantic.Field(
        ...,
        validation_alias=ANSWER_ALIASES,
    )



class AnswerDictQueryWithContextModel(AnswerDictQueryModel, WithContextMixin):
    """A query with context and an answer dictionary."""
