import typing

import pydantic
from typing_extensions import Self
from vod_datasets.rosetta import models
from vod_datasets.rosetta.adapters import adapter


class FlanQueryModel(pydantic.BaseModel):
    """A query with an alias."""

    inputs: str
    targets: str


class FlanQueryAdapter(adapter.Adapter[FlanQueryModel, models.QueryModel]):
    """An adapter for the FLAN dataset.."""

    input_model = FlanQueryModel
    output_model = models.QueryModel

    @classmethod
    def translate_row(cls: typing.Type[Self], row: dict[str, typing.Any]) -> models.QueryModel:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(
            query=m.inputs,
            answer=[m.targets],
        )
