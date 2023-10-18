import lightning as L
import torch
from lightning.fabric import wrappers as fabric_wrappers
from lightning_utilities.core.rank_zero import rank_zero_only
from vod_tools import pretty

from .base import Callback


class PprintModelStats(Callback):
    """Pretty print the model statistics."""

    @rank_zero_only
    def on_after_setup(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        if isinstance(module, fabric_wrappers._FabricModule):
            module = module.module
        pretty.pprint_parameters_stats(module, header=f"{type(module).__name__} (rank={fabric.global_rank})")
