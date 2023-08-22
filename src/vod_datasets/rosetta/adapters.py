import abc
import secrets
import typing

import datasets
import pydantic
from rich.console import Console
from typing_extensions import Self, Type, TypeVar
from vod_datasets.models import ModelType, QueryModel, QueryWithContextModel, SectionModel
from vod_datasets.rosetta.input_models import (
    AliasQueryModel,
    AliasQueryWithContextModel,
    AliasSectionModel,
    AnswerDictQueryModel,
    AnswerDictQueryWithContextModel,
    MultipleChoiceQueryModel,
)
from vod_datasets.utlis import dict_to_rich_table


class AsDict:
    """A callable that converts a pydantic model to a dict."""

    def __init__(self, fn: typing.Callable[[dict[str, typing.Any]], pydantic.BaseModel]) -> None:
        self.fn = fn

    def __call__(self, x: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Call the inner functions and dump to dict."""
        m = self.fn(x)
        return m.model_dump()


Im = TypeVar("Im", bound=pydantic.BaseModel)
Om = TypeVar("Om", bound=typing.Union[QueryModel, SectionModel, QueryWithContextModel])
Y = TypeVar("Y", bound=typing.Union[dict[str, typing.Any], datasets.Dataset, datasets.DatasetDict])


class Adapter(typing.Generic[Im, Om], abc.ABC):
    """An adapter for a dataset."""

    input_model: typing.Type[Im]
    output_model: typing.Type[Om]

    @classmethod
    def can_handle(cls: Type[Self], row: dict[str, typing.Any]) -> bool:
        """Can handle."""
        try:
            cls.input_model(**row)
            return True
        except pydantic.ValidationError:
            return False

    @classmethod
    def translate(cls: Type[Self], x: Y) -> Y:
        """Translate a row, dataset or dataset dict."""
        if isinstance(x, datasets.Dataset):
            return cls.translate_dset(x)
        if isinstance(x, datasets.DatasetDict):
            return datasets.DatasetDict({k: cls.translate_dset(v) for k, v in x.items()})  # type: ignore
        if isinstance(x, dict):
            return cls.translate_row(x).model_dump()  # type: ignore

        raise TypeError(f"Cannot translate input of type `{type(x)}`")

    @classmethod
    def translate_row(cls: typing.Type[Self], row: dict[str, typing.Any]) -> Om:
        """Placeholder for translating a row."""
        raise NotImplementedError(f"{cls.__name__} does not implement `translate_row`")

    @classmethod
    def translate_dset(cls: typing.Type[Self], dset: datasets.Dataset, **kwargs: typing.Any) -> datasets.Dataset:
        """Translating a dataset."""
        return dset.map(
            AsDict(cls.translate_row),
            remove_columns=dset.column_names,
            desc=f"Translating dataset using {cls.__name__}",
            **kwargs,
        )


class IdentityAdapter(Adapter[Om, Om]):
    """An identity adapter."""

    @classmethod
    def translate_row(cls: typing.Type[Self], row: dict[str, typing.Any]) -> Om:
        """Placeholder for translating a row."""
        return cls.output_model(**row).model_dump()

    @classmethod
    def translate_dset(cls: typing.Type[Self], dset: datasets.Dataset, **kwargs: typing.Any) -> datasets.Dataset:
        """Translating a dataset."""
        return dset


class IdentityQueryAdapter(IdentityAdapter):
    """An identity adapter for queries."""

    input_model = QueryModel
    output_model = QueryModel


class IdentitySectionAdapter(Adapter[SectionModel, SectionModel]):
    """An identity adapter for queries."""

    input_model = SectionModel
    output_model = SectionModel



class IdentityQueryWithContextAdapter(Adapter[QueryWithContextModel, QueryWithContextModel]):
    """An identity adapter for queries."""

    input_model = QueryWithContextModel
    output_model = QueryWithContextModel


class MultipleChoiceQueryAdapter(Adapter[MultipleChoiceQueryModel, QueryModel]):
    """An adapter for multiple-choice datasets."""

    input_model = MultipleChoiceQueryModel
    output_model = QueryModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> QueryModel:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(
            id=m.id,
            query=m.query,
            choices=m.choices,
            answer=[m.choices[m.answer]],
        )


class AnswerDictQueryAdapter(Adapter[AnswerDictQueryModel, QueryModel]):
    """An unified adapter for Squad and Trivia."""

    input_model = AnswerDictQueryModel
    output_model = QueryModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> QueryModel:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(
            id=m.id,
            query=m.query,
            answer=m.answer.text,
        )


class AnswerDictQueryWithContextAdapter(Adapter[AnswerDictQueryWithContextModel, QueryWithContextModel]):
    """An unified adapter for Squad and Trivia."""

    input_model = AnswerDictQueryWithContextModel
    output_model = QueryWithContextModel

    @classmethod
    def translate_row(cls: Type[Self], row: dict[str, typing.Any]) -> QueryWithContextModel:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(
            id=m.id,
            query=m.query,
            answer=m.answer.text,
            section=m.section,
        )


class AliasAdapter(Adapter[Im, Om]):
    """An adatper to rename input columns."""


    @classmethod
    def can_handle(cls: type[Self], row: dict[str, typing.Any]) -> bool:
        cls._validate_fields()
        cls._get_alias_mapping(list(row.keys()))
        return super().can_handle(row)

    @classmethod
    def translate_row(cls: typing.Type[Self], row: dict[str, typing.Any]) -> Om:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(**m.model_dump())

    @classmethod
    def translate_dset(cls: typing.Type[Self], dset: datasets.Dataset, **kwargs: typing.Any) -> datasets.Dataset:
        """Translating a dataset."""
        return dset.rename_columns(cls._get_alias_mapping(dset.column_names))
    
    @classmethod
    def _validate_fields(cls: typing.Type[Self]) -> None:
        """Validate the class."""        
        missing_in_output = set(cls.input_model.model_fields) - set(cls.output_model.model_fields)
        extra_in_output = set(cls.output_model.model_fields) - set(cls.input_model.model_fields)
        if missing_in_output or extra_in_output:
                missing_msg = f"Missing in output: `{', '.join(missing_in_output)}`" if missing_in_output else ""
                extra_msg = f"Extra in output: `{', '.join(extra_in_output)}`" if extra_in_output else ""
                error_msg = "Input and output models do not have the same fields. " + " ".join(filter(None, [missing_msg, extra_msg]))
                
                raise ValueError(error_msg)

    @classmethod
    def _get_alias_mapping(cls: typing.Type[Self], column_names: list[str]) -> dict[str, str]:
        # Build the mapping
        alias_mapping = {}
        for key, field in cls.input_model.model_fields.items():
            if field.validation_alias:
                for alias in field.validation_alias.choices:  # type: ignore
                    if alias in column_names:
                        alias_mapping[alias] = key # type: ignore
        if not alias_mapping:
            raise ValueError(f"Could not find any aliases for {cls.input_model.__name__}")
        
        return alias_mapping


class RenameQueryAdapter(AliasAdapter[AliasQueryModel, QueryModel]):
    """An adapter for multiple-choice datasets."""

    input_model = AliasQueryModel
    output_model = QueryModel


class RenameSectionAdapter(AliasAdapter[AliasSectionModel, QueryModel]):
    """An adapter for multiple-choice datasets."""

    input_model = AliasSectionModel
    output_model = QueryModel


class RenameQueryWithContextAdapter(AliasAdapter[AliasQueryWithContextModel, QueryWithContextModel]):
    """An adapter for multiple-choice datasets."""

    input_model = AliasQueryWithContextModel
    output_model = QueryWithContextModel


KNOWN_QUERY_WITH_CONTEXT_ADAPTERS: list[Type[Adapter]] = [
    IdentityQueryWithContextAdapter,
    RenameQueryWithContextAdapter,
    AnswerDictQueryWithContextAdapter,
]

KNOWN_QUERY_ADAPTERS: list[Type[Adapter]] = [
    IdentityQueryAdapter,
    RenameQueryAdapter,
    MultipleChoiceQueryAdapter,
    AnswerDictQueryAdapter,
]

KNOWN_SECTION_ADAPTERS: list[Type[Adapter]] = [
    IdentitySectionAdapter,
]

KNOWN_ADAPTERS: dict[ModelType, list[Type[Adapter]]] = {
    "query_with_context": KNOWN_QUERY_WITH_CONTEXT_ADAPTERS,
    "query": KNOWN_QUERY_ADAPTERS,
    "section": KNOWN_SECTION_ADAPTERS,
}


def get_first_row(dataset: datasets.Dataset | datasets.DatasetDict) -> dict[str, typing.Any]:
    """Get the first row of a dataset."""
    if isinstance(dataset, datasets.DatasetDict):
        # Choose a random split from the DatasetDict
        split_names = list(dataset.keys())
        random_split = secrets.choice(split_names)
        dataset = dataset[random_split]
    return dataset[0]


def find_adapter(row: dict[str, typing.Any], output: ModelType, verbose: bool = False) -> None | typing.Type[Adapter]:
    """Find an adapter for a row."""
    console = Console()
    for v in KNOWN_ADAPTERS[output]:
        if v.can_handle(row):
            translated_row = v.translate_row(row)
            if verbose:
                console.print(dict_to_rich_table(row, "Input Model"))
                console.print(dict_to_rich_table(translated_row.model_dump(), "Output Model"))
            return v

    return None
