import typing

import pydantic
from typing_extensions import Self
from vod_datasets.rosetta import models
from vod_datasets.rosetta.adapters import aliases, base


class TextToTextQueryModel(pydantic.BaseModel):
    """Text-to-Text inputs."""

    inputs: str = pydantic.Field(
        ...,
        validation_alias=aliases.INPUT_TEXTS,
    )
    targets: str = pydantic.Field(
        ...,
        validation_alias=aliases.TARGET_TEXTS,
    )


class TextToTextQueryAdapter(base.Adapter[TextToTextQueryModel, models.QueryModel]):
    """An adapter for the text-to-text datasets like FLAN."""

    input_model = TextToTextQueryModel
    output_model = models.QueryModel

    @classmethod
    def translate_row(cls: typing.Type[Self], row: dict[str, typing.Any]) -> models.QueryModel:
        """Translate a row."""
        m = cls.input_model(**row)
        return cls.output_model(
            query=m.inputs,
            answers=[m.targets],
            answer_scores=[1.0],
            retrieval_ids=[],
            retrieval_scores=[],
            subset_ids=[],
        )
