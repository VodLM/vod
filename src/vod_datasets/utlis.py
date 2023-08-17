from rich.console import Console
from rich.table import Table


def console() -> Console:
    """Get a console."""
    return Console()


# def dict_to_rich_table(data: dict, title: str) -> Table:
#     """Convert dictionary to rich table for logging."""
#     table = Table(title=title)

#     table.add_column("Key", style="cyan")
#     table.add_column("Value", style="magenta")

#     for key, value in data.items():
#         table.add_row(str(key), str(value))

#     return table


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
