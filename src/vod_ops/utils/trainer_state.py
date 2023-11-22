import typing as typ

import numpy as np
import pydantic
import vod_configs
from typing_extensions import Self


class TrainerState(pydantic.BaseModel):
    """Holds the state of the trainer."""

    class Config:
        """Pydantic configuration."""

        extra = "forbid"
        frozen = False
        from_attributes = True

    step: int
    epoch: int
    update_steps: list[int]
    must_stop: bool = False
    config: vod_configs.TrainerConfig

    @classmethod
    def from_config(cls: type[Self], config: vod_configs.TrainerConfig) -> Self:
        """Create a new `TrainerState` from a config."""
        return cls(
            step=0,
            epoch=0,
            update_steps=_infer_update_steps(config.max_steps, config.period),
            config=config,
        )

    @property
    def pidx(self) -> int:
        """Return the index of the current period."""
        for i, x in enumerate(self.update_steps):
            if self.step < x:
                return i - 1

        return len(self.update_steps) - 2

    @property
    def completed(self) -> bool:
        """Return whether the training is completed."""
        return self.step >= self.config.max_steps

    @property
    def next_period_start_step(self) -> int:
        """Return the first step of the next period."""
        return self.update_steps[self.pidx + 1]

    def get_parameters(self) -> dict[str, float]:
        """Return the parameters for a given step."""
        return {k: v(self.step) for k, v in self.config.parameters.items()}

    def iter_periods(self) -> typ.Iterator[tuple[int, int]]:
        """Iterate over the periods."""
        for i in range(len(self.update_steps) - 1):
            yield self.update_steps[i], self.update_steps[i + 1]

    def repr_update_steps(self) -> str:
        """Return a representation of the update steps."""
        return _pretty_steps(self.update_steps, max_steps=5)


def _infer_update_steps(total_number_of_steps: int, update_freq: int | list[int]) -> list[int]:
    if isinstance(update_freq, int):
        steps = [int(x) for x in np.arange(0, total_number_of_steps, update_freq)]
    elif isinstance(update_freq, list):
        if update_freq[0] != 0:
            update_freq = [0] + update_freq
        if update_freq[-1] == total_number_of_steps:
            update_freq = update_freq[:-1]
        steps = update_freq
    else:
        raise TypeError(f"Invalid type for `update_freq`: {type(update_freq)}")

    return steps + [total_number_of_steps]


def _pretty_steps(steps: list[int], max_steps: int = 5) -> str:
    steps = steps[:-1]
    if len(steps) > max_steps:
        return f"[{steps[0]}, {steps[1]}, {steps[2]}, {steps[3]}, {steps[4]} ... {steps[-1]}]"

    return str(steps)
