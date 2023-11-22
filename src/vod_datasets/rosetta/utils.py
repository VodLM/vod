import secrets
import typing

import datasets
from rich.table import Table


def dict_to_rich_table(data: dict, title: str) -> Table:
    """Convert dictionary to rich table for logging."""
    table = Table(title=title, show_header=True, header_style="bold blue")
    table.add_column("Feature", style="bold", no_wrap=False)
    table.add_column("Value", no_wrap=False)

    # For alternating row colors
    row_colors = ["black", "magenta"]

    for idx, (key, value) in enumerate(data.items()):
        # Add row with alternating colors for better readability
        table.add_row(str(key), str(value), style=row_colors[idx % 2])

    return table


def get_first_row(dataset: datasets.Dataset | datasets.DatasetDict) -> dict[str, typing.Any]:
    """Get the first row of a dataset."""
    if isinstance(dataset, datasets.DatasetDict):
        # Choose a random split from the DatasetDict
        split_names = list(dataset.keys())
        random_split = secrets.choice(split_names)
        dataset = dataset[random_split]
    return dataset[0]
