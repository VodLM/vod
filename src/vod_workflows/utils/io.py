from __future__ import annotations

import json
import pathlib

import lightning as L
import torch
from lightning.fabric import wrappers as fabric_wrappers

from .helpers import TrainerState

REL_MODEL_STATE_PATH = "model-state.ckpt"
REL_TRAINER_STATE_PATH = "trainer-state.json"


def save_training_state(
    fabric: L.Fabric,
    checkpoint_path: str,
    optimizer: None | torch.optim.Optimizer = None,
    model: None | torch.nn.Module = None,
    scheduler: None | torch.optim.lr_scheduler._LRScheduler = None,
    trainer_state: None | TrainerState = None,
) -> None:
    """Save the training state."""
    model_path = pathlib.Path(checkpoint_path) / REL_MODEL_STATE_PATH
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    fabric.save(
        model_path,
        {k: v for k, v in state.items() if v is not None},
    )

    # Save the Trainer state
    if trainer_state is not None:
        with open(pathlib.Path(checkpoint_path) / REL_TRAINER_STATE_PATH, "w") as f:
            json.dump(trainer_state.__getstate__(), f, indent=2)


def load_training_state(
    fabric: L.Fabric,
    checkpoint_path: str,
    optimizer: None | torch.optim.Optimizer = None,
    model: None | torch.nn.Module = None,
    scheduler: None | torch.optim.lr_scheduler._LRScheduler = None,
    trainer_state: None | TrainerState = None,
) -> None:
    """Load the training state."""
    model_path = pathlib.Path(checkpoint_path) / REL_MODEL_STATE_PATH
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    fabric.load(
        model_path,
        {k: v for k, v in state.items() if v is not None},
    )

    # Load the Trainer state
    if trainer_state is not None:
        with open(pathlib.Path(checkpoint_path) / REL_TRAINER_STATE_PATH, "r") as f:
            trainer_state.__setstate__(json.load(f))


def _unwrap_model(model: None | fabric_wrappers._FabricModule | torch.nn.Module) -> None | torch.nn.Module:
    if fabric_wrappers.is_wrapped(model):
        # TODO: remove this once this is handled in Fabric
        model = fabric_wrappers._unwrap_objects(model)
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            model = model.module
    return model
