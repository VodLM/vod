import os

import datasets
import fsspec
import gcsfs
from rich.console import Console
from rich.table import Table


def console() -> Console:
    """Get a console."""
    return Console()


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


def init_gcloud_filesystem() -> fsspec.AbstractFileSystem:
    """Initialize a GCS filesystem."""
    try:
        token = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError as exc:
        raise RuntimeError("Missing `GOOGLE_APPLICATION_CREDENTIALS` environment variables. ") from exc
    try:
        project = os.environ["GCLOUD_PROJECT_ID"]
    except KeyError as exc:
        raise RuntimeError("Missing `GCLOUD_PROJECT_ID` environment variables. ") from exc
    return gcsfs.GCSFileSystem(token=token, project=project)


def _fetch_queries_split(queries: datasets.DatasetDict, split: None | str) -> datasets.Dataset | datasets.DatasetDict:
    if split is None or split in {"all"}:
        return queries

    normalized_split = {
        "val": "validation",
    }.get(split, split)

    return queries[normalized_split]
