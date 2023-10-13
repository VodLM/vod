import contextlib
import functools
import typing as typ

import lightning as L
import torch
import vod_types as vt
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger

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
