from __future__ import annotations

from typing import Any

import rich


def print_metric_groups(metrics: dict[str, Any]) -> None:
    console = rich.console.Console()
    table = rich.table.Table(title="Static Validation")
    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("group", justify="left", style="green")
    table.add_column("Value", justify="right", style="magenta")

    def split_key(key: str) -> tuple[str, str]:
        *group, metric = key.split("/")
        group = "/".join(group)
        return metric, group

    metrics = [(*split_key(k), v) for k, v in metrics.items()]
    metrics = sorted(metrics, key=lambda x: (x[0], x[1]))
    prev_key = None
    for key, group, value in metrics:
        if prev_key is None:
            display_key = key
        elif key != prev_key:
            table.add_section()
            display_key = key
        else:
            display_key = ""
        value = float(value)
        table.add_row(display_key, group, f"{value:.2%}")
        prev_key = key
    console.print(table)
