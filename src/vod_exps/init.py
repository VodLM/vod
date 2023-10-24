import datetime as dt
import typing as typ

import lightning as L
import omegaconf as omg
import torch
from lightning.fabric.loggers.logger import Logger as FabricLogger
from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.strategies.fsdp import FSDPStrategy
from vod_ops.callbacks import Callback

T = typ.TypeVar("T")


def fabric(
    *args,  # noqa: ANN002
    loggers: None | typ.Iterable[FabricLogger] | typ.Mapping[str, FabricLogger] = None,
    callbacks: None | typ.Iterable[Callback] | typ.Mapping[str, Callback] = None,
    **kwargs,  # noqa: ANN003
) -> L.Fabric:
    """Initialize a fabric with the given `omegaconf`-defined loggers & callbacks."""

    def _cast_to_list(x: None | typ.Iterable[T] | typ.Mapping[str, T]) -> list[T]:
        if x is None:
            return []
        if isinstance(x, omg.DictConfig):
            x = omg.OmegaConf.to_container(x, resolve=True)  # type: ignore
        if isinstance(x, dict):
            x = x.values()
        return list(x)  # type: ignore

    return L.Fabric(
        *args,
        loggers=_cast_to_list(loggers),
        callbacks=_cast_to_list(callbacks),
        **kwargs,
    )


def ddp_strategy(
    *args,  # noqa: ANN002
    timeout: float | dt.timedelta | None = None,
    **kwargs,  # noqa: ANN003
) -> DDPStrategy:
    """Initialize a DDP strategy, parse the timeout as `dt.timedelta`."""
    if isinstance(timeout, (float, int)):
        timeout = dt.timedelta(seconds=timeout)
    return DDPStrategy(
        *args,
        timeout=timeout,
        **kwargs,
    )


def fsdp_strategy(
    *args,  # noqa: ANN002
    timeout: float | dt.timedelta | None = None,
    activation_checkpointing: None | typ.Type[torch.nn.Module] | list[typ.Type[torch.nn.Module]] = None,  # type: ignore
    **kwargs,  # noqa: ANN003
) -> FSDPStrategy:
    """Initialize a FSDP strategy, parse the timeout as `dt.timedelta`, parse layer names as python classes."""
    if isinstance(timeout, (float, int)):
        timeout = dt.timedelta(seconds=timeout)
    if activation_checkpointing is not None:
        if isinstance(activation_checkpointing, omg.ListConfig):
            activation_checkpointing = omg.OmegaConf.to_container(
                activation_checkpointing,
                resolve=True,
            )  # type: ignore
        if not isinstance(activation_checkpointing, list):
            activation_checkpointing: list[str | typ.Type] = [activation_checkpointing]  # type: ignore
        activation_checkpointing = [_import_cls(x) if isinstance(x, str) else x for x in activation_checkpointing]
    return FSDPStrategy(
        *args,
        timeout=timeout,
        activation_checkpointing=activation_checkpointing,  # type: ignore
        **kwargs,
    )


def _import_cls(x: str) -> typ.Type:
    """Import a class from a string path."""
    module_name, cls_name = x.rsplit(".", 1)
    module = __import__(module_name, fromlist=[cls_name])
    return getattr(module, cls_name)
