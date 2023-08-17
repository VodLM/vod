import abc
import secrets
import typing

import datasets
from rich.console import Console
from vod_datasets.base import QueryModel, SectionModel
from vod_datasets.utlis import dict_to_rich_table

DsetModel = typing.TypeVar("DsetModel", bound=typing.Union[QueryModel, SectionModel])


console = Console()


class Adapter(typing.Generic[DsetModel]):
    """An adapter for a dataset."""

    FIELD_MAPPING: dict[str, tuple[typing.Type, str]] = {}

    @classmethod
    def can_handle(cls: typing.Type["Adapter"], row: dict[str, typing.Any]) -> bool:
        """Can handle."""
        if all(
            key in row and isinstance(row[key], expected_type) for key, (expected_type, _) in cls.FIELD_MAPPING.items()
        ):
            return True
        return False

    @classmethod
    def map_row(cls: typing.Type["Adapter"], row: dict[str, typing.Any]) -> dict:
        """Translate a row."""
        mapped_row = {}
        for alias, (expected_type, mapped_key) in cls.FIELD_MAPPING.items():
            if alias in row and isinstance(row[alias], expected_type):
                mapped_row[mapped_key] = row[alias]
        return mapped_row

    @classmethod
    def translate(
        cls: typing.Type["Adapter"], x: datasets.Dataset | datasets.DatasetDict
    ) -> DsetModel | datasets.Dataset | datasets.DatasetDict:
        """Translate a row, dataset or dataset dict."""
        if isinstance(x, datasets.Dataset):
            return cls.translate_dset(x)
        if isinstance(x, datasets.DatasetDict):
            return datasets.DatasetDict({k: cls.translate_dset(v) for k, v in x.items()})  # type: ignore
        raise TypeError(f"Cannot translate {x} of type {type(x)}")

    @classmethod
    @abc.abstractmethod
    def translate_row(cls: typing.Type["Adapter"], row: dict[str, typing.Any]) -> dict:
        """Placeholder for translating a row."""
        ...

    @classmethod
    def translate_dset(cls: typing.Type["Adapter"], dset: datasets.Dataset, **kwargs: typing.Any) -> datasets.Dataset:
        """Translating a dataset."""
        return dset.map(cls.translate_row, remove_columns=dset.column_names, **kwargs)


class IdentityQueryAdapter(Adapter[QueryModel]):
    """An identity adapter for queries."""

    FIELD_MAPPING = {"id": (str, "id"), "query": (str, "query"), "answer": (list, "answer")}

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> dict:
        """Translate a row."""
        mapped_row = cls.map_row(row)
        return QueryModel(**mapped_row).model_dump()


class IdentitySectionAdapter(Adapter[SectionModel]):
    """An identity adapter for sections."""

    FIELD_MAPPING = {
        "content": (str, "content"),
    }

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> dict:
        """Translate a row."""
        return SectionModel(**row).model_dump()


class MmluQueryAdapter(Adapter[QueryModel]):
    """An adapter for multiple-choice datasets."""

    FIELD_MAPPING = {
        "question": (str, "query"),
        "choices": (list, "choices"),
        "answer": (int, "answer"),
    }

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> dict:
        """Translate a row."""
        row["answer"] = [row["choices"][row["answer"]]]

        mapped_row = cls.map_row(row)

        return QueryModel(**mapped_row).model_dump()


class QualityQueryAdapter(Adapter[QueryModel]):
    """An adapter for multiple-choice datasets."""

    FIELD_MAPPING = {
        "question": (str, "query"),
        "options": (list, "choices"),
        "answer": (int, "answer"),
    }

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> dict:
        """Translate a row."""
        row["answer"] = [row["options"][row["answer"]]]

        mapped_row = cls.map_row(row)

        return QueryModel(**mapped_row).model_dump()


class SquadQueryAdapter(Adapter[QueryModel]):
    """An unified adapter for Squad and Trivia."""

    FIELD_MAPPING = {"id": (str, "id"), "question": (str, "query"), "answers": (dict, "answer")}

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> dict:
        """Translate a row."""
        row["answers"] = row["answers"]["text"]

        mapped_row = cls.map_row(row)

        return QueryModel(**mapped_row).model_dump()


class TriviaQaQueryAdapter(Adapter[QueryModel]):
    """An unified adapter for Squad and Trivia."""

    FIELD_MAPPING = {"question_id": (str, "id"), "question": (str, "query"), "answer": (dict, "answer")}

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> dict:
        """Translate a row."""
        row["answer"] = [row["answer"]["value"]]
        mapped_row = cls.map_row(row)

        return QueryModel(**mapped_row).model_dump()


class MsMarcoQueryAdapter(Adapter[QueryModel]):
    """An unified adapter for Squad and Trivia."""

    FIELD_MAPPING = {"query_id": (int, "id"), "query": (str, "query"), "answers": (list, "answer")}

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> dict:
        """Translate a row."""
        row["query_id"] = str(row["query_id"])
        mapped_row = cls.map_row(row)
        return QueryModel(**mapped_row).model_dump()


class NqOpenQueryAdapter(Adapter[QueryModel]):
    """An adapter for multiple-choice datasets."""

    FIELD_MAPPING = {
        "question": (str, "query"),
        "answer": (list, "answer"),
    }

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> dict:
        """Translate a row."""
        mapped_row = cls.map_row(row)

        return QueryModel(**mapped_row).model_dump()


KNOWN_DATASETS = [
    IdentityQueryAdapter,
    IdentitySectionAdapter,
    MmluQueryAdapter,
    SquadQueryAdapter,
    MsMarcoQueryAdapter,
    QualityQueryAdapter,
    TriviaQaQueryAdapter,
    NqOpenQueryAdapter,
]


def find_adapter(dataset: typing.Union[datasets.Dataset, datasets.DatasetDict]) -> typing.Type[Adapter]:
    """Find an adapter for a row."""
    if isinstance(dataset, datasets.DatasetDict):
        # Choose a random split from the DatasetDict
        split_names = list(dataset.keys())
        random_split = secrets.choice(split_names)
        row = dataset[random_split][0]
    else:
        row = dataset[0]

    for v in KNOWN_DATASETS:
        if v.can_handle(row):
            translated_row = v.translate_row(row)
            console.print(dict_to_rich_table(row, "Original Row"))
            console.print(dict_to_rich_table(translated_row, "Translated Row"))
            return v

    raise ValueError(f"Could not find adapter for {row}")
