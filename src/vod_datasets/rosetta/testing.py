import datasets
import pydantic
from typing_extensions import Type

from .adapters import Adapter
from .utils import get_first_row


def test_parse_dataset(
    data: datasets.Dataset | datasets.DatasetDict,
    adapter_cls: Type[Adapter],
    output_model: Type[pydantic.BaseModel],
    num_proc: int = 2,
) -> None:
    """A generic test for parsing a dataset."""
    if isinstance(data, datasets.DatasetDict):
        data = data[list(data.keys())[0]]

    # Iterate over rows
    for row in data:
        if not isinstance(row, dict):
            raise NotImplementedError("Only dicts are supported")

        if not adapter_cls.can_handle(row):
            row_types = {k: type(v) for k, v in row.items()}
            raise ValueError(
                f"Cannot handle row: {row_types} using adapter `{adapter_cls}` "
                f"with input_model {adapter_cls.input_model.model_fields.keys()}"
            )

        # Test translating a row
        output = adapter_cls.translate_row(row)
        assert isinstance(output, output_model)  # noqa: S101

    # Attempt translating the entire dataset and validate the first row
    data = adapter_cls.translate(data, map_kwargs={"num_proc": num_proc})
    row = get_first_row(data)
    output_model(**row)
