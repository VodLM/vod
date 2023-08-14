import abc
import typing

import datasets
import pydantic

from .base import QueryModel, SectionModel

DsetModel = typing.TypeVar("DsetModel", bound=typing.Union[QueryModel, SectionModel])


def _rename_columns(dset: datasets.Dataset, model: typing.Type[pydantic.BaseModel]) -> datasets.Dataset:
    mapping = {k: v.alias for k, v in model.__fields__.items()}
    return dset.rename_columns(mapping)


class Adapter(typing.Generic[DsetModel], abc.ABCMeta):
    """An adapter for a dataset."""

    @classmethod
    def translate(
        cls, x: dict[str, typing.Any] | datasets.Dataset | datasets.DatasetDict
    ) -> DsetModel | datasets.Dataset | datasets.DatasetDict:
        """Translate a row, dataset or dataset dict."""
        if isinstance(x, dict):
            return cls.translate_row(x)
        if isinstance(x, datasets.Dataset):
            return cls.translate_dset(x)
        if isinstance(x, datasets.DatasetDict):
            return datasets.DatasetDict({k: cls.translate_dset(v) for k, v in x.items()})  # type: ignore
        raise TypeError(f"Cannot translate {x} of type {type(x)}")

    @abc.abstractmethod
    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> DsetModel:
        """Placeholder for translating a row."""
        ...

    @classmethod
    def translate_dset(cls, dset: datasets.Dataset, **kwargs) -> datasets.Dataset:
        """Translating a dataset."""
        return dset.map(cls.translate_row, **kwargs)


class IdentityQueryAdapter(Adapter[QueryModel]):
    """An identity adapter for queries."""

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> QueryModel:
        """Translate a row."""
        return QueryModel(**row)

    def translate_dset(self, dset: datasets.Dataset) -> datasets.Dataset:
        """Translate a dataset."""
        return dset


class IdentitySectionAdapter(Adapter[SectionModel]):
    """An identity adapter for sections."""

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> SectionModel:
        """Translate a row."""
        return SectionModel(**row)

    def translate_dset(self, dset: datasets.Dataset, **kwargs) -> datasets.Dataset:
        """Translate a dataset."""
        return dset


class FrankSectionAdapter(Adapter[SectionModel]):
    """A Frank section."""

    section: str = pydantic.Field(..., alias="content")
    kb_id: int = pydantic.Field(..., alias="knowledge_base_id")
    answer_id: int

    @classmethod
    def translate_row(cls, row: dict[str, typing.Any]) -> SectionModel:
        """Translate a row."""
        return cls(**row)

    def translate_dset(self, dset: datasets.Dataset, **kwargs) -> datasets.Dataset:
        """Translate a dataset."""
        return dset


KNOWN_DATASETS = [
    IdentityQueryAdapter,
    IdentitySectionAdapter,
    FrankSectionAdapter,
]


def find_adapter(row: dict[str, typing.Any]) -> typing.Type[Adapter]:
    """Find an adapter for a row."""
    for v in KNOWN_DATASETS:
        try:
            v.validate(row)
            return v
        except pydantic.ValidationError:
            pass

    raise ValueError(f"Could not find adapter for {row}")
