import contextlib
import dataclasses
import functools
import typing as typ

import lightning as L
import torch
import vod_configs
import vod_types as vt
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger
from typing_extensions import Self, Type
from vod_tools.misc.schedule import BaseSchedule, schedule_factory

T = typ.TypeVar("T")
D = typ.TypeVar("D", bound=vt.Sequence)
P = typ.ParamSpec("P")


def none_ok(func: typ.Callable[P, T]) -> typ.Callable[P, None | T]:
    """Decorator that allows `None` as an input."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None | T:
        if args[0] is None:
            return None
        return func(*args, **kwargs)  # type: ignore

    return wrapper


maybe_as_lazy_array = none_ok(vt.as_lazy_array)


def is_engine_enabled(parameters: None | typ.MutableMapping, engine: str) -> bool:
    """Check if an engine is enabled."""
    if parameters is None:
        return True
    return parameters.get(engine, 1.0) >= 0


def barrier_fn(name: str, fabric: L.Fabric) -> None:
    """Barrier to synchronize all processes."""
    if fabric.world_size == 1:
        return
    with contextlib.suppress(TypeError):
        logger.level("WAIT", no=12, color="<magenta>", icon="⏳")
        logger.level("PASS", no=13, color="<cyan>", icon="✅")
    logger.log("WAIT", f"barrier:wait: `{name}`")
    fabric.strategy.barrier(name)
    logger.log("PASS", f"barrier:pass: `{name}`")


@dataclasses.dataclass
class TrainerState:
    """Holds the state of the trainer."""

    step: int
    epoch: int
    pidx: int
    period: int | list[int]
    period_max_steps: None | int
    max_steps: int
    parameters: dict[str, BaseSchedule] = dataclasses.field(default_factory=dict)
    val_check_interval: int = 500
    log_interval: int = 100
    accumulate_grad_batches: int = 1
    gradient_clip_val: None | float = None
    n_max_eval: None | int = None

    def get_parameters(self) -> dict[str, float]:
        """Return the parameters for a given step."""
        return {k: v(self.step) for k, v in self.parameters.items()}

    def __getstate__(self) -> dict[str, typ.Any]:
        """Return the state of the object."""
        state = dataclasses.asdict(self)
        state["parameters"] = {k: v.model_dump() for k, v in self.parameters.items()}
        return state

    def __setstate__(self, state: dict[str, typ.Any]) -> None:
        """Set the state of the object."""
        state["parameters"] = {k: schedule_factory(**v) for k, v in state["parameters"].items()}
        self.__dict__.update(state)

    @classmethod
    def from_config(
        cls: Type[Self],
        config: vod_configs.RunConfig,
        fabric: L.Fabric,
    ) -> Self:
        """Instantiate the training state from the configuration and Lightning environment."""
        return cls(
            step=0,
            pidx=0,
            epoch=0,
            period=config.trainer.period,
            period_max_steps=None,
            max_steps=config.trainer.max_steps,
            log_interval=config.trainer.log_interval,
            val_check_interval=config.trainer.val_check_interval,
            n_max_eval=config.trainer.n_max_eval,
            accumulate_grad_batches=_infer_accumulate_grad_batches(fabric, config.batch_size),
            gradient_clip_val=config.trainer.gradient_clip_val,
            parameters=config.trainer.parameters,
        )


class _OptimizerWrapper(typ.Protocol):
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        ...


def unwrap_optimizer(optimizer: torch.optim.Optimizer | _OptimizerWrapper) -> torch.optim.Optimizer:
    """Unwrap the optimizer if it is wrapped."""
    while True:
        if isinstance(optimizer, torch.optim.Optimizer) and not type(optimizer).__name__.startswith("Fabric"):
            break
        try:
            optimizer = optimizer.optimizer  # type: ignore
        except AttributeError as exc:
            raise AttributeError(f"Could not find optimizer in `{optimizer}`") from exc

    return optimizer


unwrap_fabric_object = fabric_wrappers._unwrap_objects


def _infer_accumulate_grad_batches(fabric: L.Fabric, config: vod_configs.BatchSizeConfig) -> int:
    step_batch_size = fabric.world_size * config.per_device

    # warn if the batch size per step is larger than the effective batch size
    if step_batch_size > config.effective:
        logger.warning(
            f"Effective batch size ({config.effective}) is smaller than the batch size per step "
            f"({step_batch_size}). This will lead to a slower training."
        )
        return 1

    # accumulate gradients if the effective batch size is larger than the batch size per step
    accumulation_steps = -(-config.effective // step_batch_size)

    # warn if the effective batch size is not divisible by the batch size per step
    if config.effective % step_batch_size != 0:
        logger.warning(
            f"The effective batch size ({config.effective}) is not divisible by the batch size per step "
            f"({step_batch_size}). This will lead to a slower training."
        )

    logger.info(
        f"Using {accumulation_steps} accumulation steps. "
        f"Effective batch size: {fabric.world_size * accumulation_steps * config.per_device} "
        f"(requested={config.effective})."
    )
    return accumulation_steps
