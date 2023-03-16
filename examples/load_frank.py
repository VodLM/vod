import argparse
from functools import partial
from pathlib import Path

import datasets.fingerprint
import dill
import dotenv
import faiss
import pydantic
import lightning.pytorch as pl
import rich
import torch
import transformers
from loguru import logger
from tqdm import tqdm

from raffle_ds_research.core import builders
from raffle_ds_research.core.ml_models.simple_ranker import SimpleRanker
from raffle_ds_research.tools import index_tools, pipes, predict

dotenv.load_dotenv(Path(__file__).parent / ".predict.env")


class LoadFrankArgs(pydantic.BaseModel):
    """Arguments for the script."""

    model_name: str = "google/bert_uncased_L-4_H-256_A-4"
    language: str = "en"
    subset_name: str = "A"
    seed: int = 1
    accelerator: str = "cpu"
    batch_size: int = 10
    cache_dir: Path = Path("~/.raffle/cache/index-example/")
    factory_string: str = "IVF4,Flat"
    nprobe: int = 4
    loader_batch_size: int = 10
    num_workers: int = 2
    n_sections: int = 6
    prefetch_n_sections: int = 100
    max_pos_sections: int = 3
    use_faiss: int = 1

    @pydantic.validator("cache_dir")
    def _validate_cache_dir(cls, v):
        return Path(v).expanduser().resolve()

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        for field in cls.__fields__.values():
            parser.add_argument(f"--{field.name}", type=field.type_, default=field.default)

        args = parser.parse_args()
        return cls(**vars(args))


def run():
    """Load the Frank dataset, load a model, index Frank, and iterate over the train set.
    While iterating through the train set, check a few properties.
    """
    args = LoadFrankArgs.from_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using cache dir: {args.cache_dir.absolute()}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    logger.info(f"{type(tokenizer).__name__}: hash={datasets.fingerprint.Hasher.hash(tokenizer)}")
    builder = builders.FrankBuilder(
        language=args.language,
        subset_name=args.subset_name,
        prep_map_kwargs=dict(num_proc=4, batch_size=1000),
        tokenizer=tokenizer,
    )

    # Build the dataset
    dataset = builder()
    rich.print("=== dataset ===")
    rich.print(dataset)
    rich.print("=== dataset.corpus ===")
    rich.print(builder.get_corpus())

    # Init the model and wrap it
    model = transformers.AutoModel.from_pretrained(args.model_name)
    model = SimpleRanker(model, fields=["question", "section"], vector_name="vector")
    logger.info(f"model hash: {datasets.fingerprint.Hasher.hash(model)}")

    # Init the trainer
    logger.info(f"Instantiating the Trainer")
    trainer = pl.Trainer(accelerator=args.accelerator)

    # define the collates and dataloader args
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    question_collate = partial(pipes.torch_tokenize_collate, tokenizer=builder.tokenizer, field="question")
    logger.info(f"question_collate: hash={datasets.fingerprint.Hasher.hash(question_collate)}")
    section_collate = partial(pipes.torch_tokenize_collate, tokenizer=builder.tokenizer, field="section")
    logger.info(f"section_collate: hash={datasets.fingerprint.Hasher.hash(section_collate)}")

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
        builder.get_corpus(),
        trainer=trainer,
        cache_dir=args.cache_dir,
        model=model,
        model_output_key="section.vector",
        collate_fn=section_collate,
        loader_kwargs=loader_kwargs,
    )

    # build the faiss index and save to disk
    faiss_index = index_tools.build_index(sections_vectors, factory_string=args.factory_string)
    faiss_path = Path(args.cache_dir, "index.faiss")
    faiss.write_index(faiss_index, str(faiss_path))

    # Serve the faiss index in a separate process
    with index_tools.FaissMaster(faiss_path, args.nprobe) as faiss_master:
        faiss_client = faiss_master.get_client()

        # Instantiate the collate_fn
        loader_config = builder.collate_config(
            faiss_client=faiss_client if args.use_faiss else None,
            question_vectors=dataset_vectors["train"] if args.use_faiss else None,
            n_sections=args.n_sections,
            prefetch_n_sections=args.prefetch_n_sections,
            max_pos_sections=args.max_pos_sections,
        )
        collate_fn = builder.get_collate_fn(config=loader_config)

        # run the collate_fn on a single batch and print the result
        batch = collate_fn([dataset["train"][0], dataset["train"][1]])
        pipes.pprint_batch(batch, header="Frank - Batch")
        pipes.pprint_supervised_retrieval_batch(batch, tokenizer=builder.tokenizer, skip_special_tokens=True)

        # check if the collate_fn can be pickled
        if dill.pickles(collate_fn):
            logger.info(f"Collate is serializable. hash={datasets.fingerprint.Hasher.hash(collate_fn)}")
        else:
            logger.warning("Collate is not serializable.")

        # iterate over the batches
        loader = torch.utils.data.DataLoader(
            dataset["train"],
            collate_fn=collate_fn,
            batch_size=args.loader_batch_size,
            num_workers=args.num_workers,
        )
        for batch in tqdm(loader, desc="Iterating over batches"):
            retrieved_section_ids = batch["section.id"]
            retrieved_section_labels = batch["section.label"]
            for i in range(retrieved_section_ids.shape[0]):
                # test that there is no overlap between positive and negative sections
                ids_i = retrieved_section_ids[i, :].tolist()
                labels_i = retrieved_section_labels[i, :].tolist()
                pos_ids = [id_ for id_, label in zip(ids_i, labels_i) if label > 0]
                neg_ids = [id_ for id_, label in zip(ids_i, labels_i) if label <= 0]
                assert set.intersection(set(pos_ids), set(neg_ids)) == set()

                # test that, when a section id is set, there is only one positive section
                target_section_id = batch["question.section_id"][i]
                if target_section_id != -1:
                    assert target_section_id in pos_ids
                    assert len(pos_ids) == 1


if __name__ == "__main__":
    run()
