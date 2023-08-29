import secrets
import typing

import datasets
from rich.console import Console
from vod_datasets.rosetta.adapters import ROSETTA_ADAPTERS, Adapter
from vod_datasets.rosetta.models import ModelType
from vod_datasets.utlis import dict_to_rich_table

from .preprocessing import isolate_qa_and_sections

T = typing.TypeVar("T")


def find_adapter(row: dict[str, typing.Any], output: ModelType, verbose: bool = False) -> None | typing.Type[Adapter]:
    """Find an adapter for a row."""
    console = Console()
    for v in ROSETTA_ADAPTERS[output]:
        if v.can_handle(row):
            translated_row = v.translate_row(row)
            if verbose:
                console.print(dict_to_rich_table(row, "Input Model"))
                console.print(dict_to_rich_table(translated_row.model_dump(), "Output Model"))
            return v

    return None


def get_first_row(dataset: datasets.Dataset | datasets.DatasetDict) -> dict[str, typing.Any]:
    """Get the first row of a dataset."""
    if isinstance(dataset, datasets.DatasetDict):
        # Choose a random split from the DatasetDict
        split_names = list(dataset.keys())
        random_split = secrets.choice(split_names)
        dataset = dataset[random_split]
    return dataset[0]


class CantHandleError(ValueError):
    """Raised when input data can't be handled."""

    def __init__(
        self,
        row: dict[str, typing.Any],
        output: ModelType,
        reason: str = "",
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the error."""
        row_ = {k: type(v).__name__ for k, v in row.items()}
        message = f"Could not find an adapter for type=`{output}` and row `{row_}`. "
        message += f"Reason: {reason}"
        super().__init__(message, **kwargs)


D = typing.TypeVar("D", bound=typing.Union[datasets.Dataset, datasets.DatasetDict])


def transform(data: D, output: ModelType, num_proc: int = 4) -> D:
    """Translate a Huggingface daatset."""
    row = get_first_row(data)
    adapter: None | typing.Type[Adapter] = find_adapter(row, output="queries_with_context")

    # Process `QueriesWithContext` datasets
    if adapter is not None:
        if isinstance(data, datasets.DatasetDict):
            raise NotImplementedError("DatasetDict not supported for `QueriesWithContext`")
        data = adapter.translate(data, map_kwargs={"num_proc": num_proc})
        data_split = isolate_qa_and_sections(data, num_proc=num_proc)
        return data_split[output]  # type: ignore

    # Process `Query` or `Section` datasets
    adapter = find_adapter(row, output=output)
    if adapter is None:
        raise CantHandleError(row, output=output, reason="No matching `Adapter` could be found.")
    return adapter.translate(data, map_kwargs={"num_proc": num_proc})
