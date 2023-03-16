from __future__ import annotations

import multiprocessing

multiprocessing.set_start_method("forkserver")

import dataclasses
from typing import Optional

import pydantic

import functools
import tempfile
from pathlib import Path

import datasets
import dotenv
import faiss
import hydra
import lightning.pytorch as pl
import loguru
import rich
import rich.status
import torch
import transformers
from hydra.utils import instantiate
from lightning_fabric import seed_everything
from loguru import logger
from omegaconf import DictConfig

from raffle_ds_research.cli import utils as cli_utils
from raffle_ds_research.core.builders import FrankBuilder
from raffle_ds_research.core.ml_models import Ranker
from raffle_ds_research.tools import predict_tools, pipes, index_tools
from raffle_ds_research.tools.utils import loader_config
from raffle_ds_research.tools.utils.config import register_omgeaconf_resolvers
from raffle_ds_research.tools.utils.pretty import print_config

dotenv.load_dotenv(Path(__file__).parent / ".train.env")

register_omgeaconf_resolvers()


@hydra.main(config_path="../configs/", config_name="main", version_base="1.3")
def run(config: DictConfig):
    loguru.logger.debug(f"Multiprocessing method set to `{multiprocessing.get_start_method()}`")
    cli_utils.set_training_context()
    print_config(config)
    exp_dir = Path()
    logger.info(f"Experiment directory: {exp_dir.absolute()}")

    # Instantiate the dataset builder
    logger.info(f"Instantiating builder <{config.builder._target_}>")
    builder: FrankBuilder = instantiate(config.builder)

    # load the model
    logger.info(f"Instantiating model <{config.model._target_}>")
    seed_everything(config.seed)
    ranker: Ranker = instantiate(config.model)

    # torch 2.0 - compile the model todo
    # try:
    #     ranker = torch.compile(ranker)
    # except Exception as e:
    #     logger.warning(f"Could not compile the model: {e}")

    # Init the trainer, log the hyperparameters
    logger.info(f"Instantiating model <{config.trainer._target_}>")
    trainer: pl.Trainer = instantiate(config.trainer)
    cli_utils.log_config(trainer=trainer, config=config, exp_dir=exp_dir)

    # train the ranker
    seed_everything(config.seed, workers=True)
    train_with_index_updates(
        ranker=ranker,
        trainer=trainer,
        builder=builder,
        config=TrainWithIndexConfigs.parse(config),
    )

    # make sure to close the wandb run
    try:
        import wandb

        wandb.finish()
    except ImportError:
        ...


def _compute_vectors(
    *,
    dataset: datasets.Dataset | datasets.DatasetDict,
    model: torch.nn.Module,
    trainer: pl.Trainer,
    cache_dir: Path,
    tokenizer: transformers.PreTrainedTokenizer,
    field: str,
    loader_config: loader_config.DataLoaderConfig,
) -> predict_tools.TensorStoreFactory | dict[str, predict_tools.TensorStoreFactory]:
    output_key = {"question": "hq", "section": "hd"}[field]
    collate_fn = functools.partial(
        pipes.torch_tokenize_collate,
        tokenizer=tokenizer,
        field=field,
    )
    return predict_tools.predict(
        dataset,
        trainer=trainer,
        cache_dir=cache_dir,
        model=model,
        model_output_key=output_key,
        collate_fn=collate_fn,
        loader_kwargs=loader_config,
    )


class DefaultCollateConfig(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    n_sections: int
    max_pos_sections: int
    prefetch_n_sections: int
    sample_negatives: bool


class DefaultFaissConfig(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    factory: str
    nprobe: int
    use_faiss: bool
    update_freq: Optional[int]


@dataclasses.dataclass
class TrainWithIndexConfigs:
    train_loader: loader_config.DataLoaderConfig
    eval_loader: loader_config.DataLoaderConfig
    predict_loader: loader_config.DataLoaderConfig
    train_collate: DefaultCollateConfig
    eval_collate: DefaultCollateConfig
    faiss: DefaultFaissConfig

    @classmethod
    def parse(cls, config: DictConfig) -> "TrainWithIndexConfigs":
        # get the dataloader configs
        train_loader_config = loader_config.DataLoaderConfig(**config.loader_configs.train)
        eval_loader_config = loader_config.DataLoaderConfig(**config.loader_configs.eval)
        predict_loader_config = loader_config.DataLoaderConfig(**config.loader_configs.predict)

        # get te collate configs
        train_collate_config = DefaultCollateConfig(**config.collate_configs.train)
        eval_collate_config = DefaultCollateConfig(**config.collate_configs.eval)

        # set the faiss config
        faiss_config = DefaultFaissConfig(**config.faiss_config)

        return cls(
            train_loader=train_loader_config,
            eval_loader=eval_loader_config,
            predict_loader=predict_loader_config,
            train_collate=train_collate_config,
            eval_collate=eval_collate_config,
            faiss=faiss_config,
        )


def train_with_index_updates(
    ranker: Ranker,
    trainer: pl.Trainer,
    builder: FrankBuilder,
    config: TrainWithIndexConfigs,
) -> Ranker:
    dataset = builder()
    rich.print(dataset)

    with tempfile.TemporaryDirectory(prefix="tmp-training-") as tmpdir:
        cache_dir = Path(tmpdir)
        dataset_vectors = _compute_vectors(
            dataset=dataset,
            trainer=trainer,
            tokenizer=builder.tokenizer,
            model=ranker,
            cache_dir=cache_dir,
            field="question",
            loader_config=config.predict_loader,
        )
        sections_vectors = _compute_vectors(
            dataset=builder.get_corpus(),
            trainer=trainer,
            tokenizer=builder.tokenizer,
            model=ranker,
            cache_dir=cache_dir,
            field="section",
            loader_config=config.predict_loader,
        )

        # build the faiss index and save to disk
        faiss_index = index_tools.build_index(sections_vectors, factory_string=config.faiss.factory)
        faiss_path = Path(cache_dir, "index.faiss")
        faiss.write_index(faiss_index, str(faiss_path))

        # Serve the faiss index in a separate process
        with index_tools.FaissMaster(faiss_path, config.faiss.nprobe) as faiss_master:
            faiss_client = faiss_master.get_client()

            # Instantiate the training collate_fn
            train_collate_config = builder.collate_config(
                faiss_client=faiss_client if config.faiss.use_faiss else None,
                question_vectors=dataset_vectors["train"] if config.faiss.use_faiss else None,
                **config.train_collate.dict(),
            )
            train_collate_fn = builder.get_collate_fn(config=train_collate_config)

            # Instantiate the validation collate_fn
            val_collate_config = builder.collate_config(
                faiss_client=faiss_client if config.faiss.use_faiss else None,
                question_vectors=dataset_vectors["validation"] if config.faiss.use_faiss else None,
                **config.eval_collate.dict(),
            )
            val_collate_fn = builder.get_collate_fn(config=val_collate_config)

            # make the dataloaders
            train_loader = torch.utils.data.DataLoader(
                dataset["train"],
                collate_fn=train_collate_fn,
                **config.train_loader.dict(),
            )
            val_loader = torch.utils.data.DataLoader(
                dataset["validation"],
                collate_fn=val_collate_fn,
                **config.eval_loader.dict(),
            )

            # sample a batch and print it
            with rich.status.Status("Sampling a batch"):
                batch = next(iter(train_loader))
            pipes.pprint_batch(batch, header="Training batch")

            # validate and train the ranker
            trainer.validate(ranker, dataloaders=val_loader)
            trainer.fit(ranker, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return ranker


if __name__ == "__main__":
    run()
