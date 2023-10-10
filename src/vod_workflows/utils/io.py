import json
import pathlib

import lightning as L
import torch
from lightning.fabric import wrappers as fabric_wrappers

from .trainer_state import TrainerState

MODEL_STATE_FNAME = "model-state.ckpt"
TRAINER_STATE_PATH_FNAME = "trainer-state.json"


def save_training_state(
    fabric: L.Fabric,
    checkpoint_path: str,
    optimizer: None | torch.optim.Optimizer = None,
    model: None | torch.nn.Module = None,
    scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
    trainer_state: None | TrainerState = None,
) -> None:
    """Save the training state."""
    model_path = pathlib.Path(checkpoint_path) / MODEL_STATE_FNAME
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
        with open(pathlib.Path(checkpoint_path) / TRAINER_STATE_PATH_FNAME, "w") as f:
            f.write(trainer_state.model_dump_json(indent=2))


def load_training_state(
    fabric: L.Fabric,
    checkpoint_path: str,
    optimizer: None | torch.optim.Optimizer = None,
    module: None | torch.nn.Module = None,
    scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
    trainer_state: None | TrainerState = None,
) -> None:
    """Load the training state."""
    model_path = pathlib.Path(checkpoint_path) / MODEL_STATE_FNAME
    state = {
        "model": module,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    fabric.load(
        model_path,
        {k: v for k, v in state.items() if v is not None},
    )

    # Load the Trainer state
    if trainer_state is not None:
        with open(pathlib.Path(checkpoint_path) / TRAINER_STATE_PATH_FNAME, "r") as f:
            trainer_state = TrainerState.model_validate_json(json.load(f))


def _unwrap_model(model: None | fabric_wrappers._FabricModule | torch.nn.Module) -> None | torch.nn.Module:
    if fabric_wrappers.is_wrapped(model):
        # TODO: remove this once this is handled in Fabric
        model = fabric_wrappers._unwrap_objects(model)
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            model = model.module
    return model
