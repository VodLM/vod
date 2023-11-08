import typing

import datasets
import pydantic
from typing_extensions import Self
from vod_datasets.rosetta import models
from vod_datasets.rosetta.adapters import aliases, base


class AliasAdapter(base.Adapter[base.Im, base.Om]):
    """An adatper to rename input columns."""

    @classmethod
    def can_handle(cls: type[Self], row: dict[str, typing.Any]) -> bool:
        """Check if the adapter can handle the row."""
        cls._validate_models()
        try:
            cls._get_alias_mapping(list(row.keys()))
        except ValueError:
            return False
        return super().can_handle(row)

    @classmethod
    def translate_row(cls: typing.Type[Self], row: dict[str, typing.Any]) -> base.Om:
        """Translate a row."""
        cls._validate_models()
        m = cls.input_model(**row)
        return cls.output_model(**m.model_dump())

    @classmethod
    def translate_dset(cls: typing.Type[Self], dset: datasets.Dataset, **kwargs: typing.Any) -> datasets.Dataset:
        """Translating a dataset."""
        cls._validate_models()
        return dset.rename_columns(cls._get_alias_mapping(dset.column_names))

    @classmethod
    def _validate_models(cls: typing.Type[Self]) -> None:
        """Validate the class."""
        missing_in_output = set(cls.input_model.model_fields) - set(cls.output_model.model_fields)
        extra_in_output = set(cls.output_model.model_fields) - set(cls.input_model.model_fields)
        if missing_in_output or extra_in_output:
            missing_msg = f"Missing in output: `{', '.join(missing_in_output)}`" if missing_in_output else ""
            extra_msg = f"Extra in output: `{', '.join(extra_in_output)}`" if extra_in_output else ""
            error_msg = "Input and output models do not have the same fields. " + " ".join(
                filter(None, [missing_msg, extra_msg])
            )

            raise ValueError(error_msg)

    @classmethod
    def _get_alias_mapping(cls: typing.Type[Self], column_names: list[str]) -> dict[str, str]:
        """Get the mapping from input keys to output keys."""
        alias_mapping = {}
        for key, field in cls.input_model.model_fields.items():
            if field.validation_alias:
                for alias in field.validation_alias.choices:  # type: ignore
                    if alias in column_names:
                        alias_mapping[alias] = key  # type: ignore
        if not alias_mapping:
            raise ValueError(f"Could not find any aliases for {cls.input_model.__name__}")
        return alias_mapping


class AliasSectionModel(models.SectionModel):
    """A query with an alias."""

    id: str = pydantic.Field(
        ...,
        validation_alias=aliases.SECTION_ID_ALIASES,
    )
    content: str = pydantic.Field(
        ...,
        validation_alias=aliases.SECTION_ALIASES,
    )
    title: str = pydantic.Field(
        default=None,
        validation_alias=aliases.TITLES_ALIASES,
    )


class RenameSectionAdapter(AliasAdapter[AliasSectionModel, models.SectionModel]):
    """An adapter to rename sections."""

    input_model = AliasSectionModel
    output_model = models.SectionModel
