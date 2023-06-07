from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Optional

import hydra
import loguru
import omegaconf
import torch
from hydra.utils import instantiate
from lightning_fabric import seed_everything
from loguru import logger
from omegaconf import DictConfig

try:
    multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    loguru.logger.debug("Could not set multiprocessing start method to `forkserver`")

import lightning.pytorch as pl  # noqa: E402
from lightning.pytorch import utilities as pl_utils  # noqa: E402

from raffle_ds_research.cli import utils as cli_utils  # noqa: E402
from raffle_ds_research.core import workflows  # noqa: E402
from raffle_ds_research.core.ml import Ranker  # noqa: E402
from raffle_ds_research.tools.utils.config import register_omgeaconf_resolvers  # noqa: E402
from raffle_ds_research.tools.utils.pretty import print_config  # noqa: E402

# richuru.install(rich_traceback=False)  # <- setup rich logging with loguru
register_omgeaconf_resolvers()


class ModelGenerator:
    """Initialize a ranking model from a config."""

    def __init__(self, model_config: omegaconf.DictConfig, seed: Optional[int] = None, compile: bool = False):
        self.model_config = model_config
        self.seed = seed
        self.compile = compile

    def __call__(self) -> Ranker:
        """Instantiate the model."""
        if self.seed is not None:
            seed_everything(self.seed)
        ranker: Ranker = instantiate(self.model_config)
        if self.compile:
            return torch.compile(ranker)
        return ranker


@hydra.main(config_path="../configs/", config_name="main", version_base="1.3")
def run(config: DictConfig) -> None:
    """Train a ranker for a retrieval task."""
    logger.debug(f"Setting environment variables from config: {config.env}")
    os.environ.update({k: str(v) for k, v in config.env.items()})
    logger.debug(f"Multiprocessing method set to `{multiprocessing.get_start_method()}`")  # type: ignore
    logger.debug(
        f"Distributed: RANK={os.environ.get('GLOBAL_RANK', None)}, " f"WORLD_SIZE={os.environ.get('WORLD_SIZE', None)}"
    )
    cli_utils.set_training_context()
    pl_utils.rank_zero_only(print_config)(config)
    exp_dir = Path()
    logger.info(f"Experiment directory: {exp_dir.absolute()}")

    # load the model
    logger.info(f"Instantiating model generator <{config.model._target_}>")
    seed_everything(config.seed)
    ranker_generator = ModelGenerator(model_config=config.model, seed=config.seed, compile=config.compile)

    # Init the trainer, log the hyperparameters
    logger.info(f"Instantiating model <{config.trainer._target_}>")
    trainer: pl.Trainer = instantiate(config.trainer)
    pl_utils.rank_zero_only(cli_utils.log_config)(ranker=ranker_generator(), config=config, exp_dir=exp_dir)

    logger.info(f"Training the ranker with seed={config.seed}")
    seed_everything(config.seed, workers=True)
    workflows.train_with_index_updates(
        trainer=trainer,
        ranker_generator=ranker_generator,
        config=config,
    )


if __name__ == "__main__":
    try:
        run()
        logger.info(f"Success. Experiment logged to {Path().absolute()}")
    except Exception as exc:
        loguru.logger.warning(f"Failure. Experiment logged to {Path().absolute()}")
        raise exc

    # make sure to close the wandb run
    try:
        import wandb

        wandb.finish()
    except ImportError:
        ...
