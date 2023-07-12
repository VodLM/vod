from __future__ import annotations

import multiprocessing
import os
import pathlib
import sys
from pathlib import Path

import hydra
import lightning as L
import loguru
import omegaconf
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

try:
    multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    loguru.logger.debug("Could not set multiprocessing start method to `forkserver`")

from src import vod_configs, vod_models, vod_workflows
from src.vod_tools.misc.omega_utils import register_omgeaconf_resolvers
from src.vod_tools.misc.pretty import print_config
from src.vod_workflows.cli import utils as cli_utils

register_omgeaconf_resolvers()


def _is_gloabl_zero() -> bool:
    """Check if the current process is the global zero."""
    return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0"


@hydra.main(config_path=vod_configs.hyra_conf_path(), config_name="main", version_base="1.3")
def run(config: DictConfig) -> None:
    """Train a ranker for a retrieval task."""
    if config.load_from is not None:
        logger.info(f"Loading checkpoint from `{config.load_from}`")
        checkpoint_path = config.load_from
        cfg_path = pathlib.Path(config.load_from, "config.yaml")
        config = omegaconf.OmegaConf.load(cfg_path)  # type: ignore
    else:
        checkpoint_path = None

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
    if config.seed is not None:
        fabric.seed_everything(config.seed)
    ranker: vod_models.Ranker = instantiate(config.model)

    # Log config & setup logger
    _customize_logger(fabric=fabric)
    if fabric.is_global_zero:
        cli_utils.log_config(
            config=config,
            exp_dir=exp_dir,
            extras={"meta": cli_utils._get_ranker_meta_data(ranker)},
            fabric=fabric,
        )

    # Train the model
    logger.info(f"Training the ranker with seed={config.seed}")
    vod_workflows.train_with_index_updates(
        fabric=fabric,
        ranker=ranker,
        config=config,
        load_from=checkpoint_path,
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

    if not fabric.is_global_zero:
        # change the log level to avoid spamming the console
        logger.configure(
            handlers=[{"sink": sys.stderr, "level": "WARNING", "format": logger_format}],
            extra={"rank": 1 + fabric.global_rank, "world_size": fabric.world_size},
        )
