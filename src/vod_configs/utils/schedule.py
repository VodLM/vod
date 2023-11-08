import math
import typing as typ

from typing_extensions import Self, Type
from vod_configs.utils.base import StrictModel

ParameterScheduleModes = typ.Literal["constant", "linear", "step", "exponential"]


class ParameterSchedule(StrictModel):
    """Defines a parameter schedule."""

    mode: ParameterScheduleModes
    value: float = 1.0
    start: float = 0
    period: int = int(1e9)
    offset: int = 0

    def __call__(self, step: float) -> float:  # noqa: PLR0911
        """Return the value of the parameter at the given step."""
        if self.mode == "constant":
            return self.value

        if self.mode == "linear":
            if step < self.offset:
                return self.start
            if step >= self.offset + self.period:
                return self.value
            return self.start + (self.value - self.start) * (step - self.offset) / self.period

        if self.mode == "step":
            if step < self.period:
                return self.start
            return self.value

        if self.mode == "exponential":
            if step < self.offset:
                return self.start
            return self.start + (self.value - self.start) * (1 - math.exp(-1.0 * (step - self.offset) / self.period))

        raise ValueError(f"Unknown mode: {self.mode}")

    @classmethod
    def parse(cls: Type[Self], config_or_value: str | float | int | dict[str, typ.Any]) -> Self:
        """Parse a parameter schedule."""
        if isinstance(config_or_value, (str, float, int)):
            return cls(mode="constant", value=float(config_or_value))  # type: ignore

        return cls(**config_or_value)
