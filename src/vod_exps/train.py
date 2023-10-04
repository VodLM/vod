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

import vod_configs
import vod_models
from vod_exps import recipes
from vod_exps import utils as cli_utils
from vod_tools import pretty

from .hydra import hyra_conf_path, register_omgeaconf_resolvers

register_omgeaconf_resolvers()


def _is_gloabl_zero() -> bool:
    """Check if the current process is the global zero."""
    return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0"


@hydra.main(config_path=hyra_conf_path(), config_name="train", version_base="1.3")
def run(hydra_config: DictConfig) -> None:
    """Train a ranker for a retrieval task."""
    if hydra_config.load_from is not None:
        logger.info(f"Loading checkpoint from `{hydra_config.load_from}`")
        checkpoint_dir = hydra_config.load_from
        cfg_path = pathlib.Path(hydra_config.load_from, "config.yaml")
        hydra_config = omegaconf.OmegaConf.load(cfg_path)  # type: ignore

    else:
        checkpoint_dir = None

    if _is_gloabl_zero():
        pretty.pprint_config(hydra_config)

    logger.debug(f"Setting environment variables from config: {hydra_config.env}")
    os.environ.update({k: str(v) for k, v in hydra_config.env.items()})
    cli_utils.set_training_context()
    exp_dir = Path()
    logger.info(f"Experiment directory: {exp_dir.absolute()}")

    # Init the Fabric, log the hyperparameters
    logger.info(f"Instantiating Lighning's Fabric <{hydra_config.fabric._target_}>")
    omegaconf.OmegaConf.to_container(hydra_config, resolve=True)
    fabric: L.Fabric = instantiate(hydra_config.fabric)
    fabric.launch()

    # Init the model
    logger.info(f"Instantiating model <{hydra_config.model._target_}>")
    if hydra_config.seed is not None:
        fabric.seed_everything(hydra_config.seed)
    ranker: vod_models.Ranker = instantiate(hydra_config.model)

    if _is_gloabl_zero():
        pretty.pprint_parameters_stats(ranker, header="Model Parameters (init)")

    # Log config & setup logger
    _customize_logger(fabric=fabric)
    if fabric.is_global_zero:
        cli_utils.log_config(
            config=hydra_config,
            exp_dir=exp_dir,
            extras={"meta": cli_utils._get_ranker_meta_data(ranker)},
            fabric=fabric,
        )

    # Parse the configuration
    config = vod_configs.RunConfig.parse(hydra_config)

    # Train the model
    logger.info(f"Training the ranker with seed={hydra_config.seed}")
    recipes.periodic_training(
        fabric=fabric,
        ranker=ranker,
        config=config,
        resume_from_checkpoint=checkpoint_dir,
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
