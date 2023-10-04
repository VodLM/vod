import dataclasses
import typing as typ

from rich.console import Console
from rich.table import Table


@dataclasses.dataclass
class TableRow:
    """A row in the table."""

    key: str
    group: str
    value: float


def pprint_metric_dict(
    metrics: dict[str, typ.Any],
    header: None | str = None,
    console: None | Console = None,
) -> None:
    """Nicely print a dictionary of metrics using `rich.table.Table`."""
    console = console or Console()
    table = Table(title=header or "Static Validation")
    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("group", justify="left", style="green")
    table.add_column("Value", justify="right", style="magenta")

    def _make_row(key: str, value: typ.Any) -> TableRow:  # noqa: ANN401
        *group, metric = key.split("/")
        group = "/".join(group)
        return TableRow(key=metric, group=group, value=float(value))

    metrics_rows = [_make_row(k, v) for k, v in metrics.items()]
    metrics_rows = sorted(metrics_rows, key=lambda x: (x.key, x.group))
    prev_key = None
    for row in metrics_rows:
        if prev_key is None:
            display_key = row.key
        elif row.key != prev_key:
            table.add_section()
            display_key = row.key
        else:
            display_key = ""
        table.add_row(display_key, row.group, f"{row.value:.2%}")
        prev_key = row.key
    console.print(table)
