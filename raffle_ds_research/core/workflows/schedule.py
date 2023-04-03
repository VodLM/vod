from typing import Literal, Protocol

import pydantic


class Schedule(Protocol):
    def __call__(self, step: float, start: float, end: float, period: int, offset: int = 0) -> float:
        ...


def LinearSchedule(step: float, start: float, end: float, period: int, offset: int = 0) -> float:
    if step < offset:
        return start
    elif step >= offset + period:
        return end
    else:
        return start + (end - start) * (step - offset) / period


SCHEDULES = {
    "linear": LinearSchedule,
}


class ScheduleConfig(pydantic.BaseModel):
    class Config:
        extra: pydantic.Extra.forbid

    type: Literal["constant", "linear"]
    start: float
    end: float
    period: int
    offset: int = 0

    def __call__(self, step: float) -> float:
        schedule: Schedule = SCHEDULES[self.type]
        return schedule(step, self.start, self.end, self.period, self.offset)
