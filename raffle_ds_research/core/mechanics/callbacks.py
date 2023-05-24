from __future__ import annotations

from typing import Any

import lightning as L
import omegaconf
import rich
from lightning.pytorch import callbacks as pl_callbacks

from raffle_ds_research.core.workflows.utils.schedule import schedule_factory


class DecayLrOnFitCallaback(pl_callbacks.Callback):
    def __init__(self, schedule: dict[str, Any] | omegaconf.DictConfig) -> None:
        super().__init__()
        if isinstance(schedule, (float, int)):
            self.schedule = schedule_factory(mode="constant", value=schedule)
        else:
            self.schedule = schedule_factory(**schedule)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        lr_master = self.schedule(trainer.global_step)
        rich.print(f"[magenta]### Setting learning rate to {lr_master}")
        # # set the learning rate
        # for optimizer in trainer.optimizers:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr_master

        # for scheduler in trainer.lr_schedulers:
        #     scheduler["scheduler"].step(trainer.global_step)
