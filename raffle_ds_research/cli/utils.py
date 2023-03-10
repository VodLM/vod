from __future__ import annotations

from pathlib import Path
from typing import Optional

import datasets
import pytorch_lightning as pl
import torch
import transformers
from omegaconf import OmegaConf

from raffle_ds_research.utils.config import config_to_flat_dict


def _set_context():
    """Set the general context for torch, datasets, etc."""
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    torch.set_float32_matmul_precision("medium")


def _log_config(trainer, config, exp_dir):
    """Log the config as hyperparameters and save it locally and on MLFlow."""
    trainer.logger.log_hyperparams(config_to_flat_dict(config))
    mlflow_logger = _fetch_mlflow_logger(trainer.loggers)
    config_path = Path(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))
    if mlflow_logger:
        mlflow_logger._mlflow_client.log_artifact(run_id=mlflow_logger.run_id, local_path=config_path)


def _fetch_mlflow_logger(loggers: list[pl.loggers.base.LightningLoggerBase]) -> Optional[pl.loggers.MLFlowLogger]:
    for logger in loggers:
        if isinstance(logger, pl.loggers.mlflow.MLFlowLogger):
            return logger
