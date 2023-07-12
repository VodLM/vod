from __future__ import annotations

from typing import Any

import lightning as L
import omegaconf
from lightning.pytorch import callbacks as pl_callbacks

from raffle_ds_research.core.workflows.utils.schedule import schedule_factory


class DecayLrOnFitCallaback(pl_callbacks.Callback):
    """Decay the main learning rate on fit start."""

    def __init__(self, schedule: dict[str, Any] | omegaconf.DictConfig) -> None:
        super().__init__()
        if isinstance(schedule, (float, int)):
            self.schedule = schedule_factory(mode="constant", value=schedule)
        else:
            self.schedule = schedule_factory(**schedule)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # noqa: ARG002
        """Set the learning rate on fit start."""
        lr_master = self.schedule(trainer.global_step)  # noqa: F841
        raise NotImplementedError("TODO: set lr_master on pl_module")
