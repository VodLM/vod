import lightning as L
import torch
from lightning.fabric import wrappers as fabric_wrappers
from vod_tools import pretty

from .base import Callback


class PprintModelStats(Callback):
    """Pretty print the model statistics."""

    def on_after_setup(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        if fabric.is_global_zero:
            if isinstance(module, fabric_wrappers._FabricModule):
                module = module.module
            pretty.pprint_parameters_stats(module, header=f"{type(module).__name__} (rank={fabric.global_rank})")
