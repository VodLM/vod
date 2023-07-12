from __future__ import annotations

import lightning as L
import torch
from lightning.fabric import wrappers as fabric_wrappers

from .helpers import TrainerState


def save_training_state(
    fabric: L.Fabric,
    checkpoint_path: str,
    optimizer: None | torch.optim.Optimizer = None,
    model: None | torch.nn.Module = None,
    scheduler: None | torch.optim.lr_scheduler._LRScheduler = None,
    trainer_state: None | TrainerState = None,
) -> None:
    """Save the training state."""
    model = _unwrap_model(model)

    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "trainer_state": trainer_state,
    }
    fabric.save(
        checkpoint_path,
        {k: v for k, v in state.items() if v is not None},
    )


def load_training_state(
    fabric: L.Fabric,
    checkpoint_path: str,
    optimizer: None | torch.optim.Optimizer = None,
    model: None | torch.nn.Module = None,
    scheduler: None | torch.optim.lr_scheduler._LRScheduler = None,
    trainer_state: None | TrainerState = None,
) -> None:
    """Load the training state."""
    model = _unwrap_model(model)
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "trainer_state": trainer_state,
    }
    fabric.load(
        checkpoint_path,
        {k: v for k, v in state.items() if v is not None},
    )


def _unwrap_model(model: None | fabric_wrappers._FabricModule | torch.nn.Module) -> None | torch.nn.Module:
    if fabric_wrappers.is_wrapped(model):
        # TODO: remove this once this is handled in Fabric
        model = fabric_wrappers._unwrap_objects(model)
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            model = model.module
    return model
