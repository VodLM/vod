from __future__ import annotations

import datasets
import hydra
import pytorch_lightning as pl
import rich
import torch
import transformers
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from raffle_ds_research.datasets.builders import FrankBuilder
from raffle_ds_research.ml_models.ranker import Ranker
from raffle_ds_research.tools.utils.config import register_omgeaconf_resolvers
from raffle_ds_research.tools.utils.pretty import print_config

register_omgeaconf_resolvers()


@hydra.main(config_path="../configs/", config_name="main", version_base="1.3")
def run(config: DictConfig):
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    print_config(config)
    with open("config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(config))

    # Instantiate the dataset builder
    logger.info(f"Instantiating builder <{config.builder._target_}>")
    builder: FrankBuilder = instantiate(config.builder)

    # build the Frank dataset, get the collate_fn
    logger.info(f"Building the Frank ({builder.split}) dataset..")
    seed_everything(config.seed)
    dataset = builder()
    collate_fn = builder.get_collate_fn()
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

    # Trainer
    logger.info(f"Instantiating model <{config.trainer._target_}>")
    trainer: pl.Trainer = instantiate(config.trainer)

    # Init the data loaders
    train_loader = DataLoader(dataset["train"], collate_fn=collate_fn, **config.train_loader_kwargs)
    val_loader = DataLoader(dataset["validation"], collate_fn=collate_fn, **config.eval_loader_kwargs)

    # train the ranker
    seed_everything(config.seed, workers=True)
    trainer.fit(ranker, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    run()
