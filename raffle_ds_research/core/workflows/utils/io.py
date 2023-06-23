from typing import Optional

import lightning as L
import torch

from .support import TrainerState


def save_training_state(
    fabric: L.Fabric,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    trainer_state: TrainerState,
) -> None:
    """Save the training state."""
    fabric.save(
        checkpoint_path,
        {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "trainer_state": trainer_state,
        },
    )


def load_training_state(
    fabric: L.Fabric,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    trainer_state: TrainerState,
) -> None:
    """Load the training state."""
    fabric.load(
        checkpoint_path,
        {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "trainer_state": trainer_state,
        },
    )
