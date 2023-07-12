from __future__ import annotations

import functools
import math
import pathlib
from pathlib import Path
from typing import Any

import dill
import dotenv
import faiss
import lightning.pytorch as pl
import pydantic
import rich
import torch
import transformers
from loguru import logger
from raffle_ds_research.core import mechanics
from raffle_ds_research.core.ml.monitor import Monitor, RetrievalMetricCollection
from raffle_ds_research.core.ml.simple_ranker import SimpleRanker
from raffle_ds_research.tools import arguantic, index_tools, pipes, predict
from raffle_ds_research.tools.index_tools import faiss_tools
from raffle_ds_research.utils.pretty import print_metric_groups
from tqdm import tqdm

dotenv.load_dotenv(Path(__file__).parent / ".predict.env")


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    dset_name: str = "frank.A.en.pos"
    model_name: str = "google/bert_uncased_L-4_H-256_A-4"
    seed: int = 1
    accelerator: str = "cpu"
    batch_size: int = 10
    cache_dir: Path = Path("~/.raffle/cache/index-example/")
    factory_string: str = "IVF4,Flat"
    nprobe: int = 4
    loader_batch_size: int = 10
    num_workers: int = 0
    n_sections: int = 50
    prefetch_n_sections: int = 300
    max_pos_sections: int = 10
    faiss_weight: float = 1.0
    bm25_weight: float = 1.0

    @pydantic.validator("cache_dir")
    def _validate_cache_dir(cls, v: str | Path) -> pathlib.Path:  # noqa: N805
        return pathlib.Path(v).expanduser().resolve()


MAX_BATCHES = 10


def run() -> None:  # noqa: PLR0915
    """Load retrieval dataset, load a model, index, and iterate over the train set."""
    args = Args.parse()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using cache dir: {args.cache_dir.absolute()}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    logger.info(f"{type(tokenizer).__name__}: hash={pipes.fingerprint(tokenizer)}")
    builder = mechanics.DatasetFactory.from_name(
        name=args.dset_name,
        prep_map_kwargs={"num_proc": 4, "batch_size": 1000},
        tokenizer=tokenizer,
    )

    # Build the dataset
    dataset = builder()
    rich.print("=== dataset ===")
    rich.print(dataset)
    rich.print("=== dataset.corpus ===")
    sections = builder.get_sections()
    rich.print(sections)

    # Init the model and wrap it
    model = transformers.AutoModel.from_pretrained(args.model_name)
    model = SimpleRanker(model, fields=["question", "section"], vector_name="vector")
    logger.info(f"model hash: {pipes.fingerprint(model)}")

    # Init the trainer
    logger.info("Instantiating the Trainer")
    trainer = pl.Trainer(accelerator=args.accelerator)

    # define the collates and dataloader args
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": True}
    question_collate = functools.partial(
        pipes.torch_tokenize_collate,
        tokenizer=tokenizer,
        prefix_key="question.",
        max_length=512,
        truncation=True,
    )
    logger.info(f"question_collate: hash={pipes.fingerprint(question_collate)}")
    section_collate = functools.partial(
        pipes.torch_tokenize_collate,
        tokenizer=tokenizer,
        prefix_key="section.",
        max_length=512,
        truncation=True,
    )
    logger.info(f"section_collate: hash={pipes.fingerprint(section_collate)}")

    # Predict - compute the vectors for the question datasets and the sections
    dataset_vectors = predict(
        dataset,
        trainer=trainer,
        cache_dir=args.cache_dir,
        model=model,
        model_output_key="question.vector",
        collate_fn=question_collate,
        loader_kwargs=loader_kwargs,
    )
    sections_vectors = predict(
        sections,
        trainer=trainer,
        cache_dir=args.cache_dir,
        model=model,
        model_output_key="section.vector",
        collate_fn=section_collate,
        loader_kwargs=loader_kwargs,
    )

    # build the faiss index and save to disk
    faiss_index = faiss_tools.build_faiss_index(sections_vectors, factory_string=args.factory_string)
    faiss_path = Path(args.cache_dir, "index.faiss")
    faiss.write_index(faiss_index, str(faiss_path))

    # Serve the faiss index in a separate process
    with index_tools.FaissMaster(faiss_path, args.nprobe) as faiss_master:
        sections_content = (r["content"] for r in sections)
        sections_label = (r["kb_id"] for r in sections)
        with index_tools.Bm25Master(sections_content, groups=sections_label, input_size=len(sections)) as bm25_master:
            faiss_client = faiss_master.get_client()
            bm25_client = bm25_master.get_client()

            # Instantiate the collate_fn
            loader_config = builder.collate_config(
                question_vectors=dataset_vectors["train"],
                clients=[
                    {"name": "faiss", "client": faiss_client, "weight": args.faiss_weight},
                    {"name": "bm25", "client": bm25_client, "weight": args.bm25_weight},
                ],
                n_sections=args.n_sections,
                prefetch_n_sections=args.prefetch_n_sections,
                max_pos_sections=args.max_pos_sections,
            )
            collate_fn = builder.get_collate_fn(config=loader_config)

            # run the collate_fn on a single batch and print the result
            batch = collate_fn([dataset["train"][0], dataset["train"][1]])
            pipes.pprint_batch(batch, header="Frank - Batch")
            pipes.pprint_retrieval_batch(batch, tokenizer=builder.tokenizer, skip_special_tokens=True)

            # check if the collate_fn can be pickled
            if dill.pickles(collate_fn):
                logger.info(f"Collate is serializable. hash={pipes.fingerprint(collate_fn)}")
            else:
                logger.warning("Collate is not serializable.")

            # iterate over the batches
            loader = torch.utils.data.DataLoader(
                dataset["train"],
                collate_fn=collate_fn,
                batch_size=args.loader_batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )

            monitor = Monitor(
                metrics=RetrievalMetricCollection(
                    metrics=["ndcg", "mrr", "hitrate@01", "hitrate@03", "hitrate@10", "hitrate@30"]
                ),
                splits=["score", "bm25", "faiss"],
            )

            hits: dict[str, Any] = {"score": [], "bm25": [], "faiss": []}
            for idx, batch in enumerate(tqdm(loader, desc="Iterating over batches")):
                if idx > MAX_BATCHES:
                    break

                monitor.update_from_retrieval_batch(batch, field="section")
                for key, hit in hits.items():
                    logits = batch[f"section.{key}"]
                    targets = batch["section.label"]
                    for logits_i, target_i in zip(logits, targets):
                        logits_i_ = logits_i.masked_fill(target_i.isnan(), -math.inf)
                        target_i_ = target_i.masked_fill(logits_i.isinf(), 0)
                        ids = torch.argsort(logits_i_, descending=True)
                        target_i_ = target_i_[ids]
                        r = {f"hitrate@{k}": (target_i_[:k] > 0).any().item() for k in [1, 3, 10, 30]}
                        hit.append(r)

                retrieved_section_ids = batch["section.id"]
                retrieved_section_labels = batch["section.label"]
                for i in range(retrieved_section_ids.shape[0]):
                    # test that there is no overlap between positive and negative sections
                    ids_i = retrieved_section_ids[i, :].tolist()
                    labels_i = retrieved_section_labels[i, :].tolist()
                    pos_ids = [id_ for id_, label in zip(ids_i, labels_i) if label > 0]
                    neg_ids = [id_ for id_, label in zip(ids_i, labels_i) if label <= 0]
                    if set.intersection(set(pos_ids), set(neg_ids)) != set():
                        raise ValueError("Overlap between positive and negative sections.")

            metrics = monitor.compute()
            print_metric_groups(metrics)

            # show the python metrics for comparison
            py_metrics = {}
            for key, key_hits in hits.items():
                keys = key_hits[0].keys()
                py_metrics[key] = {k: sum([r[k] for r in key_hits]) / len(hit) for k in keys}
            rich.print(py_metrics)


if __name__ == "__main__":
    run()
