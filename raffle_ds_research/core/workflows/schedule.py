from __future__ import annotations

from typing import Literal, Protocol

import pydantic


class Schedule(Protocol):
    """Defines a parameter schedule."""

    def __call__(self, step: float, start: float, end: float, period: int, offset: int = 0) -> float:
        """Return the value of the parameter at the given step."""
        ...


def linear_schedule(step: float, start: float, end: float, period: int, offset: int = 0) -> float:
    """Linear schedule."""
    if step < offset:
        return start
    if step >= offset + period:
        return end

    return start + (end - start) * (step - offset) / period


SCHEDULES = {
    "linear": linear_schedule,
}


class ScheduleConfig(pydantic.BaseModel):
    """Configures a parameter schedule."""

    class Config:
        """Pydantic config."""

        extra: pydantic.Extra.forbid

    type: Literal["constant", "linear"]
    start: float
    end: float
    period: int
    offset: int = 0

    def __call__(self, step: float) -> float:
        """Return the value of the parameter at the given step."""
        schedule: Schedule = SCHEDULES[self.type]
        return schedule(step, self.start, self.end, self.period, self.offset)
