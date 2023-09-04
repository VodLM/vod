from __future__ import annotations

from typing import Any, Literal, Optional, Type

import pydantic


class BaseSchedule(pydantic.BaseModel):
    """Defines a parameter schedule."""

    class Config:
        """Pydantic config."""

        extra = "forbid"
        allow_mutation = False

    mode: str
    value: float = 1.0
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


SCHEDULES: list[Type[BaseSchedule]] = [
    LinearSchedule,
    StepSchedule,
    ConstantSchedule,
]
SCHEDULES_MAP = {s.model_fields["mode"].default: s for s in SCHEDULES}


def schedule_factory(*, mode: Literal["constant", "linear", "step"] = "constant", **kwargs: Any) -> BaseSchedule:
    """Return a schedule factor."""
    return SCHEDULES_MAP[mode](**kwargs)
