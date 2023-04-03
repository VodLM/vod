# pylint: disable=unspecified-encoding
from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any

import datasets
import lightning.pytorch as pl
import loguru
import omegaconf
import torch
import transformers

from raffle_ds_research.utils.config import config_to_flat_dict


def _do_nothing(*args: Any, **kwargs: Any) -> None:
    """Do nothing."""


def _return_same(value: Any) -> Any:
    """Return the same value."""
    return value


def _cast_hps(value: Any) -> str:
    """Cast a value to a string."""
    formatter = {
        numbers.Number: _return_same,
        str: _return_same,
        Path: lambda x: str(x.absolute()),
    }.get(type(value), str)
    return formatter(value)


def set_training_context() -> None:
    """Set the general context for torch, datasets, etc."""
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    torch.set_float32_matmul_precision("medium")

    # torch.multiprocessing.set_sharing_strategy("file_system")


def log_config(trainer: pl.Trainer, config: omegaconf.DictConfig, exp_dir: Path) -> None:
    """Log the config as hyperparameters and save it locally."""
    config_path = Path(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(omegaconf.OmegaConf.to_yaml(config, resolve=True))

    # log he config to wandb
    try:
        # pylint: disable=import-outside-toplevel
        import wandb

        try:
            flat_config = config_to_flat_dict(config, sep="/")
            flat_config = {k: _cast_hps(v) for k, v in flat_config.items()}
            wandb.config.update(flat_config)
        except wandb.errors.Error:
            loguru.logger.debug("Could not log config to wandb")
    except ImportError:
        ...
