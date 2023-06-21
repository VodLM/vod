from typing import Any

from rich import progress, text


class ProcessingSpeedColumn(progress.ProgressColumn):
    """Renders human readable processing speed."""

    def render(self, task: progress.Task) -> text.Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return text.Text("?", style="progress.data.speed")
        return text.Text(f"{speed:.2f} batch/s", style="progress.data.speed")


class BatchProgressBar(progress.Progress):
    """Progress bar for batch processing."""

    def __init__(self, **kwarg: Any):
        columns = [
            progress.TextColumn("[bold blue]{task.description}", justify="right"),
            progress.BarColumn(bar_width=None),
            ProcessingSpeedColumn(),
            "•",
            progress.TimeRemainingColumn(),
            "•",
            progress.TextColumn("{task.fields[info]}", justify="left"),
        ]
        super().__init__(*columns, **kwarg)
