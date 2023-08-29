import typing

import datasets
from typing_extensions import Self
from vod_datasets.rosetta import models
from vod_datasets.rosetta.adapters import adapter


class IdentityAdapter(adapter.Adapter[adapter.Om, adapter.Om]):
    """An identity adapter."""

    @classmethod
    def translate_row(cls: typing.Type[Self], row: dict[str, typing.Any]) -> adapter.Om:
        """Placeholder for translating a row."""
        return cls.output_model(**row)

    @classmethod
    def translate_dset(cls: typing.Type[Self], dset: datasets.Dataset, **kwargs: typing.Any) -> datasets.Dataset:
        """Translating a dataset."""
        return dset


class IdentityQueryAdapter(IdentityAdapter[models.QueryModel]):
    """An identity adapter for queries."""

    input_model = models.QueryModel
    output_model = models.QueryModel


class IdentitySectionAdapter(IdentityAdapter[models.SectionModel]):
    """An identity adapter for queries."""

    input_model = models.SectionModel
    output_model = models.SectionModel


class IdentityQueryWithContextAdapter(IdentityAdapter[models.QueryWithContextsModel]):
    """An identity adapter for queries."""

    input_model = models.QueryWithContextsModel
    output_model = models.QueryWithContextsModel
