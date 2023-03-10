from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
import rich
import torch
from hydra.utils import instantiate
from lightning_fabric import seed_everything
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from raffle_ds_research.cli.utils import _log_config, _set_context
from raffle_ds_research.core.builders import FrankBuilder
from raffle_ds_research.core.ml_models import Ranker
from raffle_ds_research.tools.utils.config import register_omgeaconf_resolvers
from raffle_ds_research.tools.utils.pretty import print_config

register_omgeaconf_resolvers()


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
    logger.info(f"Building `{config.builder.name}` dataset..")
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
    _log_config(trainer, config, exp_dir)

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


if __name__ == "__main__":
    run()
