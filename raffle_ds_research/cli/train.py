from __future__ import annotations
from typing import Iterable, Optional
import omegaconf
import multiprocessing

from pathlib import Path

import dotenv
import hydra
import torch

import loguru
from hydra.utils import instantiate
from lightning_fabric import seed_everything
from loguru import logger
from omegaconf import DictConfig
import richuru

multiprocessing.set_start_method("forkserver")

import lightning.pytorch as pl  # noqa: E402

from raffle_ds_research.cli import utils as cli_utils  # noqa: E402
from raffle_ds_research.core import dataset_builders, workflows  # noqa: E402
from raffle_ds_research.core.ml_models import Ranker  # noqa: E402
from raffle_ds_research.tools.utils.config import register_omgeaconf_resolvers  # noqa: E402
from raffle_ds_research.tools.utils.pretty import print_config  # noqa: E402

richuru.install()  # <- setup rich logging with loguru

dotenv.load_dotenv(Path(__file__).parent / ".train.env")

register_omgeaconf_resolvers()


class ModelGenerator:
    """Initialize a ranking model from a config."""

    def __init__(self, model_config: omegaconf.DictConfig, seed: Optional[int] = None, compile: bool = False):
        self.model_config = model_config
        self.seed = seed
        self.compile = compile

    def __call__(self) -> Ranker:
        if self.seed is not None:
            seed_everything(self.seed)
        ranker: Ranker = instantiate(self.model_config)
        if self.compile:
            ranker = torch.compile(ranker)
        return ranker


@hydra.main(config_path="../configs/", config_name="main", version_base="1.3")
def run(config: DictConfig) -> None:
    loguru.logger.debug(f"Multiprocessing method set to `{multiprocessing.get_start_method()}`")  # type: ignore
    cli_utils.set_training_context()
    print_config(config)
    exp_dir = Path()
    logger.info(f"Experiment directory: {exp_dir.absolute()}")

    # Instantiate the dataset builder
    logger.info(f"Instantiating builder <{config.builder._target_}>")
    builder: dataset_builders.RetrievalBuilder = instantiate(config.builder)

    # load the model
    logger.info(f"Instantiating model <{config.model._target_}>")
    seed_everything(config.seed)
    ranker_generator = ModelGenerator(model_config=config.model, seed=config.seed, compile=config.compile)

    # Init the trainer, log the hyperparameters
    logger.info(f"Instantiating model <{config.trainer._target_}>")
    trainer: pl.Trainer = instantiate(config.trainer)
    cli_utils.log_config(ranker=ranker_generator(), config=config, exp_dir=exp_dir)

    logger.info(f"Training the ranker with seed={config.seed}")
    seed_everything(config.seed, workers=True)
    benchmark_builders = list(_fetch_benchmark_builders(ref_builder=builder, config=config.benchmark))
    workflows.train_with_index_updates(
        ranker_generator=ranker_generator,
        trainer=trainer,
        builder=builder,
        config=config,
        monitor=instantiate(config.monitor),
        benchmark_builders=benchmark_builders,
        benchmark_on_init=config.benchmark.on_init,
    )


def _fetch_benchmark_builders(
    ref_builder: dataset_builders.RetrievalBuilder, config: omegaconf.DictConfig
) -> Iterable[dataset_builders.RetrievalBuilder]:
    for builder_config in config.builders:
        overrides = omegaconf.OmegaConf.to_container(builder_config, resolve=True)
        new_config = ref_builder.config.copy(update=overrides)
        new_builder = dataset_builders.RetrievalBuilder.from_name(**new_config.dict())
        yield new_builder


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
