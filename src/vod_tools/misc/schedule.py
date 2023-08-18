from __future__ import annotations

from typing import Any, Literal, Optional

import pydantic


class BaseSchedule(pydantic.BaseModel):
    """Defines a parameter schedule."""

    class Config:
        """Pydantic config."""

        extra = "forbid"
        frozen = False

    value: float
    start: float
    period: int
    offset: int = 0

    def __call__(self, step: float) -> float:
        """Return the value of the parameter at the given step."""
        ...


class LinearSchedule(BaseSchedule):
    """Defines a linear schedule."""

    mode: str = "linear"

    def __call__(self, step: float) -> float:
        """Return the value of the parameter at the given step."""
        if step < self.offset:
            return self.start
        if step >= self.offset + self.period:
            return self.value

        return self.start + (self.value - self.start) * (step - self.offset) / self.period


class StepSchedule(BaseSchedule):
    """Defines a step schedule."""

    mode: str = "step"

    def __call__(self, step: float) -> float:
        """Return the value of the parameter at the given step."""
        if step < self.period:
            return self.start
        return self.value


class ConstantSchedule(BaseSchedule):
    """Defines a constant schedule."""

    start: Optional[float] = None
    period: Optional[int] = None
    offset: Optional[int] = None
    mode: str = "constant"

    def __call__(self, step: float) -> float:  # noqa: ARG002
        """Return the value of the parameter at the given step."""
        return self.value


def schedule_factory(*, mode: Literal["constant", "linear"], **kwargs: Any) -> BaseSchedule:
    """Return a schedule factor."""
    if mode == "constant":
        return ConstantSchedule(**kwargs)
    if mode == "linear":
        return LinearSchedule(**kwargs)
    if mode == "step":
        return StepSchedule(**kwargs)
    raise ValueError(f"Invalid mode: {mode}")
