from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import datasets
import faiss
import hydra
import pytorch_lightning as pl
import rich
import torch
import transformers
from hydra.utils import instantiate
from lightning_fabric import seed_everything
from loguru import logger
from omegaconf import DictConfig

from raffle_ds_research.cli.utils import _set_context
from raffle_ds_research.core.builders import FrankBuilder
from raffle_ds_research.core.ml_models import Ranker
from raffle_ds_research.tools import index_tools, predict
from raffle_ds_research.tools.index_tools import RetrievalBatch
from raffle_ds_research.tools.index_tools.client import FaissClient, FaissMaster
from raffle_ds_research.tools.index_tools.vector_handler import VectorHandler, vector_handler
from raffle_ds_research.tools.utils.config import register_omgeaconf_resolvers
from raffle_ds_research.tools.utils.pretty import print_config

register_omgeaconf_resolvers()


def collate_tokenized_field(
    egs: list[dict],
    *,
    tokenizer: transformers.PreTrainedTokenizer,
    field: str = "question",
) -> dict[str, torch.Tensor]:
    keys = ["input_ids", "attention_mask"]
    keys = [f"{field}.{k}" for k in keys]
    if any(k not in egs[0] for k in keys):
        raise ValueError(f"Missing keys {keys} in examples. Found keys: {egs[0].keys()}")
    egs = [{str(k).replace(f"{field}.", ""): eg[k] for k in keys} for eg in egs]
    output = tokenizer.pad(egs, return_tensors="pt")
    return {f"{field}.{k}": v for k, v in output.items()}


def search_index(
    _: Any,
    idx: list[int],
    *,
    faiss_client: FaissClient,
    q_vectors: VectorHandler,
    top_k: int = 10,
) -> dict[str, Any]:
    query_vec = q_vectors[idx]
    results = faiss_client.search(query_vec, top_k=top_k)
    return results.to_dict()


@hydra.main(config_path="../raffle_ds_research/configs/", config_name="main", version_base="1.3")
def run(config: DictConfig):
    _set_context()
    print_config(config)
    exp_dir = Path()
    cache_dir = Path(config.sys.cache_dir, "examples-index/")
    logger.info(f"Experiment directory: {exp_dir.absolute()}")
    logger.info(f"Cache directory: {cache_dir.absolute()}")

    # Instantiate the dataset builder
    logger.info(f"Instantiating builder <{config.builder._target_}>")
    builder: FrankBuilder = instantiate(config.builder)

    # build the Frank dataset, get the collate_fn
    logger.info(f"Building `{config.builder.name}` dataset..")
    seed_everything(config.seed)
    dataset = builder()
    rich.print(dataset)
    sections = builder.get_corpus()
    if sections is None:
        raise ValueError(f"No corpus found for builder {type(builder)}")
    rich.print(sections)

    # load the model
    logger.info(f"Instantiating model <{config.model._target_}>")
    seed_everything(config.seed)
    ranker: Ranker = instantiate(config.model)
    ranker.eval()
    ranker.freeze()

    # Init the trainer
    logger.info(f"Instantiating model <{config.trainer._target_}>")
    trainer: pl.Trainer = instantiate(config.trainer)

    # compute the vectors
    dataset_vectors = predict(
        dataset,
        trainer=trainer,
        cache_dir=cache_dir,
        model=ranker,
        model_output_key="hq",
        collate_fn=partial(collate_tokenized_field, tokenizer=builder.tokenizer, field="question"),
        loader_kwargs=config.predict_loader_kwargs,
    )
    rich.print(dataset_vectors)
    sections_vectors = predict(
        sections,
        trainer=trainer,
        cache_dir=cache_dir,
        model=ranker,
        model_output_key="hd",
        collate_fn=partial(collate_tokenized_field, tokenizer=builder.tokenizer, field="section"),
        loader_kwargs=config.predict_loader_kwargs,
    )
    rich.print(sections_vectors)

    # create the faiss index
    index_path = Path(config.sys.cache_dir, "examples-index/index.faiss")
    index: faiss.Index = index_tools.build_index(sections_vectors, factory_string="Flat")
    faiss.write_index(index, str(index_path.absolute()))

    ids = [0, 101, 202]
    with FaissMaster(index_path, nprobe=8, logging_level="critical") as faiss_master:
        faiss_client = faiss_master.get_client()
        for split, store in dataset_vectors.items():
            q_vectors = vector_handler(store)[ids]
            rich.print(q_vectors)
            results = faiss_client.search(q_vectors, top_k=3)
            rich.print(results)

            for i, result in enumerate(results):
                question = dataset[split][ids[i]]
                record = _make_record(i, split, question, result, sections)
                rich.print(record)

    # call within multiprocessing
    logger.info("Testing multiprocessing: search sections for the training set..")
    search_pipe = partial(
        search_index,
        faiss_client=faiss_client,
        q_vectors=dataset_vectors["train"],
        top_k=3,
    )
    map_kwargs = dict(batched=True, with_indices=True, batch_size=32, num_proc=4)
    mapped_train_set = dataset["train"].map(search_pipe, **map_kwargs)
    rich.print(mapped_train_set)


def _make_record(i: int, split: str, question: dict, result: RetrievalBatch, sections: datasets.Dataset) -> dict:
    record = {
        "split": split,
        "i": i,
        **{k: v for k, v in question.items() if k in ["question", "knowledge_base_id"]},
        "sections": [],
    }
    for j, (s, k) in enumerate(zip(result.scores, result.indices)):
        sec = sections[int(k)]
        record["sections"].append(
            {
                "index": s,
                "score": k,
                **{k: v for k, v in sec.items() if k in ["title", "content", "knowledge_base_id"]},
            }
        )
    return record


if __name__ == "__main__":
    run()
