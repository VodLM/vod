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
from loguru import logger
from omegaconf import DictConfig

from raffle_ds_research.cli.utils.misc import _get_ranker_meta_data

try:
    multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    loguru.logger.debug("Could not set multiprocessing start method to `forkserver`")
from raffle_ds_research.cli import utils as cli_utils  # noqa: E402
from raffle_ds_research.core import workflows  # noqa: E402
from raffle_ds_research.core.ml import Ranker  # noqa: E402
from raffle_ds_research.tools.utils.config import register_omgeaconf_resolvers  # noqa: E402
from raffle_ds_research.tools.utils.pretty import print_config  # noqa: E402

register_omgeaconf_resolvers()


def _instantiate_ranker(
    model_config: omegaconf.DictConfig,
    seed: Optional[int] = None,
    fabric: Optional[L.Fabric] = None,
    compile: bool = False,
) -> Ranker:
    """Initialize a ranking model from a config."""
    if seed is not None:
        logger.info(f"Setting seed={seed}")
        if fabric is None:
            L.seed_everything(seed)
        else:
            fabric.seed_everything(seed)
    ranker: Ranker = instantiate(model_config)
    if compile:
        return torch.compile(ranker)

    return ranker


def _is_gloabl_zero() -> bool:
    """Check if the current process is the global zero."""
    return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0"


@hydra.main(config_path="../configs/", config_name="main", version_base="1.3")
def run(config: DictConfig) -> None:
    """Train a ranker for a retrieval task."""
    if _is_gloabl_zero():
        print_config(config)

    logger.debug(f"Setting environment variables from config: {config.env}")
    os.environ.update({k: str(v) for k, v in config.env.items()})
    cli_utils.set_training_context()
    exp_dir = Path()
    logger.info(f"Experiment directory: {exp_dir.absolute()}")

    # Init the Fabric, log the hyperparameters
    logger.info(f"Instantiating Lighning's Fabric <{config.fabric._target_}>")
    omegaconf.OmegaConf.to_container(config, resolve=True)
    fabric: L.Fabric = instantiate(config.fabric)
    fabric.launch()

    # Init the model
    logger.info(f"Instantiating model <{config.model._target_}>")
    ranker = _instantiate_ranker(model_config=config.model, seed=config.seed, compile=config.compile, fabric=fabric)

    # Log config & setup logger
    _customize_logger(fabric=fabric)
    if fabric.is_global_zero:
        cli_utils.log_config(config=config, exp_dir=exp_dir, extras={"meta": _get_ranker_meta_data(ranker)})

    # Train the model
    logger.info(f"Training the ranker with seed={config.seed}")
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


def _customize_logger(fabric: L.Fabric) -> None:
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
