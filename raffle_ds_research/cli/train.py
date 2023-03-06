from __future__ import annotations

from pathlib import Path
from typing import Optional

import datasets
import hydra
import pytorch_lightning as pl
import rich
import torch
import transformers
from hydra.utils import instantiate
from lightning_fabric import seed_everything
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from raffle_ds_research.core.builders import FrankBuilder
from raffle_ds_research.core.ml_models import Ranker
from raffle_ds_research.tools.utils.config import register_omgeaconf_resolvers
from raffle_ds_research.tools.utils.pretty import print_config
from raffle_ds_research.utiils.config import config_to_flat_dict

register_omgeaconf_resolvers()


def fetch_mlflow_logger(loggers: list[pl.loggers.base.LightningLoggerBase]) -> Optional[pl.loggers.MLFlowLogger]:
    for logger in loggers:
        if isinstance(logger, pl.loggers.mlflow.MLFlowLogger):
            return logger


@hydra.main(config_path="../configs/", config_name="main", version_base="1.3")
def run(config: DictConfig):
    _set_context()
    print_config(config)
    exp_dir = Path()
    logger.info(f"Experiment directory: {exp_dir.absolute()}")

    # Instantiate the dataset builder
    logger.info(f"Instantiating builder <{config.builder._target_}>")
    builder: FrankBuilder = instantiate(config.builder)

    # build the Frank dataset, get the collate_fn
    logger.info(f"Building the Frank ({builder.split}) dataset..")
    seed_everything(config.seed)
    dataset = builder()
    rich.print(dataset)

    # load the model
    logger.info(f"Instantiating model <{config.model._target_}>")
    seed_everything(config.seed)
    ranker: Ranker = instantiate(config.model)

    # torch 2.0 - compile the model
    try:
        ranker = torch.compile(ranker)
    except Exception as e:
        logger.warning(f"Could not compile the model: {e}")

    # Init the trainer, log the hyperparameters
    logger.info(f"Instantiating model <{config.trainer._target_}>")
    trainer: pl.Trainer = instantiate(config.trainer)
    log_config(trainer, config, exp_dir)

    # Init the data loaders
    train_loader = DataLoader(
        dataset["train"],
        collate_fn=builder.get_collate_fn(split="train"),
        **config.train_loader_kwargs,
    )
    val_loader = DataLoader(
        dataset["validation"],
        collate_fn=builder.get_collate_fn(split="validation"),
        **config.eval_loader_kwargs,
    )

    # train the ranker (evaluate first)
    seed_everything(config.seed, workers=True)
    trainer.validate(ranker, dataloaders=val_loader)
    trainer.fit(ranker, train_dataloaders=train_loader, val_dataloaders=val_loader)


def _set_context():
    """Set the general context for torch, datasets, etc."""
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    torch.set_float32_matmul_precision("medium")


def log_config(trainer, config, exp_dir):
    """Log the config as hyperparameters and save it locally and on MLFlow."""
    trainer.logger.log_hyperparams(config_to_flat_dict(config))
    mlflow_logger = fetch_mlflow_logger(trainer.loggers)
    config_path = Path(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))
    if mlflow_logger:
        mlflow_logger._mlflow_client.log_artifact(run_id=mlflow_logger.run_id, local_path=config_path)


if __name__ == "__main__":
    run()
