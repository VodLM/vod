import typing as typ

import torch
from vod_workflows.utils.trainer_state import TrainerState

T = typ.TypeVar("T")


def format_metric_value(
    v: T,
) -> T:
    """Format a metric value so it is safe for logging.."""
    if isinstance(v, torch.Tensor):
        return v.detach().mean().cpu()

    return v


def format_pbar_info(
    state: TrainerState,
    train_metrics: None | dict[str, typ.Any] = None,
    eval_metrics: None | dict[str, typ.Any] = None,
    keys: None | list[str] = None,
) -> str:
    """Format the metrics for the progress bar."""
    keys = keys or ["loss"]
    desc = (
        f"{state.step}/{state.next_period_start_step} ({state.config.max_steps}) "
        f"• epoch={1+state.epoch} "
        f"• grad-acc={state.config.accumulate_grad_batches}"
    )
    if train_metrics or eval_metrics:
        suppl = []
        if train_metrics is not None:
            for k in keys:
                if k in train_metrics:
                    suppl.append(f"train/{k}={train_metrics[k]:.2f}")

        if eval_metrics is not None:
            for k in keys:
                if k in eval_metrics:
                    suppl.append(f"val/{k}={eval_metrics[k]:.2f}")

        desc = f"[yellow]{' '.join(suppl)}[/yellow] • {desc}"

    return desc


class RunningAverage:
    """Running average for metrics."""

    _sums: dict[str, float | torch.Tensor]
    _counts: dict[str, int]

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset the metric."""
        self._sums = {}
        self._counts = {}

    @staticmethod
    def _sum(x: float | torch.Tensor) -> float | torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.sum()
        return x

    @staticmethod
    def _numel(x: float | torch.Tensor) -> int:
        if isinstance(x, torch.Tensor):
            return x.numel()
        return 1

    def add(self, key: str, value: float | torch.Tensor) -> None:
        """Update the metric with a new value."""
        if key not in self._sums:
            self._sums[key] = self._sum(value)
            self._counts[key] = self._numel(value)
        else:
            self._sums[key] += self._sum(value)
            self._counts[key] += self._numel(value)

    def average(self, key: str) -> float | torch.Tensor:
        """Return the average of the metric."""
        return self._sums[key] / self._counts[key]

    def update(self, data: dict[str, float | torch.Tensor]) -> None:
        """Update the metric with a new batch of data."""
        for key, value in data.items():
            self.add(key, value)

    def get(self) -> dict[str, float | torch.Tensor]:
        """Get the running average for all metrics."""
        return {key: self.average(key) for key in self._sums}

    def __str__(self):
        return ", ".join(f"{key}: {self.average(key):.2f}" for key in self._sums)
