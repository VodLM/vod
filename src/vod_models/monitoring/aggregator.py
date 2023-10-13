import abc

import torch


class Agregator(abc.ABC, torch.nn.Module):
    """Aggregates merics."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the metric stats."""

    @abc.abstractmethod
    def update(self, values: torch.Tensor) -> None:
        """Update the metrics stats."""

    @abc.abstractmethod
    def get(self) -> torch.Tensor:
        """Return the metric value averaged over all updates."""

    @abc.abstractmethod
    def all_reduce(self) -> None:
        """Synchronize all processes by summing stats."""


class MeanAggregator(Agregator):
    """Computes the mean."""

    _total: torch.Tensor
    _count: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self._total = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        self._count = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        self.reset()

    def reset(self) -> None:
        """Reset the metric stats."""
        self._total.data.fill_(0.0)
        self._count.data.fill_(0.0)

    def update(self, values: torch.Tensor) -> None:
        """Update the metrics stats."""
        values = values.detach()
        self._total += values.sum()
        self._count += values.numel()

    def get(self) -> torch.Tensor:
        """Return the metric value averaged over all updates."""
        return (self._total / self._count).mean()

    def all_reduce(self) -> None:
        """Synchronize all processes by summing stats."""
        torch.distributed.all_reduce(self._total.data, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(self._count.data, op=torch.distributed.ReduceOp.SUM)
