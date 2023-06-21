from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import lightning as L
import loguru
import omegaconf
import torch
from hydra.utils import instantiate
from lightning.fabric.loggers import Logger as FabricLogger
from lightning_fabric import seed_everything
from loguru import logger
from omegaconf import DictConfig

try:
    multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    loguru.logger.debug("Could not set multiprocessing start method to `forkserver`")

from lightning.pytorch import utilities as pl_utils  # noqa: E402

from raffle_ds_research.cli import utils as cli_utils  # noqa: E402
from raffle_ds_research.core import workflows  # noqa: E402
from raffle_ds_research.core.ml import Ranker  # noqa: E402
from raffle_ds_research.tools.utils.config import register_omgeaconf_resolvers  # noqa: E402
from raffle_ds_research.tools.utils.pretty import print_config  # noqa: E402

# richuru.install(rich_traceback=False)  # <- setup rich logging with loguru
register_omgeaconf_resolvers()


def _instantiate_ranker(
    model_config: omegaconf.DictConfig,
    seed: Optional[int] = None,
    compile: bool = False,
) -> Ranker:
    """Initialize a ranking model from a config."""
    if seed is not None:
        logger.info(f"Setting seed={seed}")
        L.seed_everything(seed)
    ranker: Ranker = instantiate(model_config)
    if compile:
        return torch.compile(ranker)

    return ranker


@hydra.main(config_path="../configs/", config_name="main", version_base="1.3")
def run(config: DictConfig) -> None:
    """Train a ranker for a retrieval task."""
    logger.debug(f"Setting environment variables from config: {config.env}")
    os.environ.update({k: str(v) for k, v in config.env.items()})
    cli_utils.set_training_context()
    pl_utils.rank_zero_only(print_config)(config)
    exp_dir = Path()
    logger.info(f"Experiment directory: {exp_dir.absolute()}")

    # Setup WandB and log config
    logger.info(f"Instantiating logger <{list(config.loggers.keys())}>")
    loggers: dict[str, FabricLogger] = {key: instantiate(lcfg) for key, lcfg in config.loggers.items()}

    # Init the model
    logger.info(f"Instantiating model generator <{config.model._target_}>")
    ranker = _instantiate_ranker(model_config=config.model, seed=config.seed, compile=config.compile)

    # Init the Fabric, log the hyperparameters
    logger.info(f"Instantiating Lighning's Fabric <{config.fabric._target_}>")
    omegaconf.OmegaConf.to_container(config, resolve=True)
    fabric: L.Fabric = instantiate(config.fabric, loggers=list(loggers.values()), _convert_="object")
    fabric.launch()

    # log config & setup logger
    pl_utils.rank_zero_only(cli_utils.log_config)(config=config, exp_dir=exp_dir)
    _configure_logger(fabric=fabric)

    logger.info(f"Training the ranker with seed={config.seed}")
    seed_everything(config.seed, workers=True)
    workflows.train_with_index_updates(
        fabric=fabric,
        ranker=ranker,
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


def _configure_logger(fabric: L.Fabric) -> None:
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<yellow>rank</yellow>:<yellow>{extra[rank]}/{extra[world_size]}</yellow> - <level>{message}</level>"
    )
    if fabric.world_size == 1:
        return

    logger.configure(extra={"rank": 1 + fabric.global_rank, "world_size": fabric.world_size})  # Default values
    logger.remove()
    logger.add(sys.stderr, format=logger_format)
