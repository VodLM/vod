# pylint: disable=too-many-instance-attributes,fixme

from __future__ import annotations

import copy
import dataclasses
import functools
import pathlib
import tempfile
from pathlib import Path
from typing import Any, Optional

import datasets
import faiss
import loguru
import rich
import torch
import transformers
from lightning import pytorch as pl

from raffle_ds_research.core.dataset_builders import retrieval_builder
from raffle_ds_research.core.dataset_builders.retrieval_collate import SearchClientConfig
from raffle_ds_research.core.ml_models import Ranker
from raffle_ds_research.core.workflows.config import MultiIndexConfig
from raffle_ds_research.tools import index_tools, pipes, predict_tools
from raffle_ds_research.tools.index_tools import faiss_tools
from raffle_ds_research.tools.pipes.utils.misc import keep_only_columns
from raffle_ds_research.tools.utils.loader_config import DataLoaderConfig


@dataclasses.dataclass
class Vectors:
    """Stores the vectors for the questions/dataset and the sections."""

    dataset: dict[str, predict_tools.TensorStoreFactory]
    sections: Optional[predict_tools.TensorStoreFactory] = None


class IndexManager:
    """Takes care of building the dataset, computing the vectors and spinning up the indexes."""

    _dataset: Optional[datasets.DatasetDict] = None
    _tmpdir: Optional[tempfile.TemporaryDirectory] = None
    _faiss_master: Optional[index_tools.FaissMaster] = None
    _bm25_master: Optional[index_tools.Bm25Master] = None
    _clients: Optional[list[SearchClientConfig]] = None
    _vectors: Optional[Vectors] = None
    _cache_dir: Optional[Path] = None

    def __init__(
        self,
        *,
        ranker: Ranker,
        trainer: pl.Trainer,
        builder: retrieval_builder.RetrievalBuilder,
        index_cfg: MultiIndexConfig,
        loader_cfg: DataLoaderConfig,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.ranker = ranker
        self.trainer = trainer
        self.builder = builder
        self.index_cfg = index_cfg
        self.loader_cfg = loader_cfg
        self._cache_dir = cache_dir

    @property
    def clients(self) -> list[SearchClientConfig]:
        """Returns the list of search clients."""
        if self._clients is None:
            raise ValueError("IndexManager has not been initialized.")
        return copy.copy(self._clients)

    @property
    def vectors(self) -> Vectors:
        """Return a copy of the vectors (questions+sections)."""
        if self._vectors is None:
            raise ValueError("IndexManager has not been initialized.")
        return copy.copy(self._vectors)

    @property
    def dataset(self) -> datasets.DatasetDict:
        """Return a copy of the dataset (questions)."""
        if self._dataset is None:
            raise ValueError("IndexManager has not been initialized.")
        return copy.copy(self._dataset)

    def __enter__(self) -> "IndexManager":
        """Build the dataset and spin up the indexes."""
        self._dataset = self.builder()
        rich.print(self._dataset)

        # create a temporary working directory if not provided
        if self._cache_dir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="tmp-training-")
            cache_dir = self._tmpdir.__enter__()
        else:
            cache_dir = self._cache_dir

        loguru.logger.info(f"Setting up index at `{cache_dir}`")
        is_rank_zero = self.trainer.global_rank == 0

        # init the bm25 index
        bm25_weight = self.index_cfg.bm25.get_weight(self.trainer.global_step)
        if bm25_weight >= 0:
            corpus = self.builder.get_corpus()
            corpus = keep_only_columns(corpus, [self.index_cfg.bm25.indexed_key, self.index_cfg.bm25.label_key])
            index_name = f"index-{corpus._fingerprint}"
            self._bm25_master = index_tools.Bm25Master(
                texts=(row[self.index_cfg.bm25.indexed_key] for row in corpus),
                labels=(row[self.index_cfg.bm25.label_key] for row in corpus)
                if self.index_cfg.bm25.use_labels
                else None,
                input_size=len(corpus),
                index_name=index_name,
                exist_ok=True,
                persistent=True,
                skip_setup=not is_rank_zero,
            )
            bm25_master = self._bm25_master.__enter__()
        else:
            bm25_master = None

        # init the faiss index
        faiss_weight = self.index_cfg.faiss.get_weight(self.trainer.global_step)
        if faiss_weight >= 0:
            # compute the vectors for the questions and sections
            self._vectors = self._compute_vectors(cache_dir)

            # build the faiss index and save to disk
            faiss_path = Path(cache_dir, "index.faiss")
            if is_rank_zero:
                # build the faiss index, but only on the main process
                faiss_index = faiss_tools.build_index(
                    self._vectors.sections,
                    factory_string=self.index_cfg.faiss.factory,
                )
                faiss.write_index(faiss_index, str(faiss_path))

            # spin up the faiss server
            self.trainer.strategy.barrier("faiss_index_setup")
            self._faiss_master = index_tools.FaissMaster(
                faiss_path,
                self.index_cfg.faiss.nprobe,
                skip_setup=not is_rank_zero,
            )
            faiss_master = self._faiss_master.__enter__()

        else:
            self._vectors = Vectors(dataset={})
            faiss_master = None

        # register clients
        self._clients = []
        if faiss_master is not None:
            self._clients.append(
                SearchClientConfig(
                    name="faiss",
                    client=faiss_master.get_client(),
                    weight=faiss_weight,
                    label_key=self.index_cfg.faiss.label_key,
                )
            )
        if bm25_master is not None:
            self._clients.append(
                SearchClientConfig(
                    name="bm25",
                    client=bm25_master.get_client(),
                    weight=bm25_weight,
                    label_key=self.index_cfg.bm25.label_key,
                )
            )

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ANN401
        """Clean up everything."""
        self._clients = None
        self._vectors = None
        self._dataset = None
        if self._faiss_master is not None:
            self._faiss_master.__exit__(exc_type, exc_val, exc_tb)
            self._faiss_master = None

        if self._cache_dir is None:
            self._tmpdir.__exit__(exc_type, exc_val, exc_tb)
            self._tmpdir = None

    def _compute_vectors(self, tmpdir: pathlib.Path) -> Vectors:
        dataset_vectors = _compute_dataset_vectors(
            dataset=self._dataset,
            trainer=self.trainer,
            tokenizer=self.builder.tokenizer,
            model=self.ranker,
            cache_dir=tmpdir,
            field="question",
            loader_config=self.loader_cfg,
        )
        sections_vectors = _compute_dataset_vectors(
            dataset=self.builder.get_corpus(),
            trainer=self.trainer,
            tokenizer=self.builder.tokenizer,
            model=self.ranker,
            cache_dir=tmpdir,
            field="section",
            loader_config=self.loader_cfg,
        )
        return Vectors(dataset=dataset_vectors, sections=sections_vectors)


def _compute_dataset_vectors(
    *,
    dataset: datasets.Dataset | datasets.DatasetDict,
    model: torch.nn.Module,
    trainer: pl.Trainer,
    cache_dir: Path,
    tokenizer: transformers.PreTrainedTokenizer,
    field: str,
    max_length: int = 512,  # todo: don't hardcode this
    loader_config: DataLoaderConfig,
) -> predict_tools.TensorStoreFactory | dict[str, predict_tools.TensorStoreFactory]:
    output_key = {"question": "hq", "section": "hd"}[field]
    collate_fn = functools.partial(
        pipes.torch_tokenize_collate,
        tokenizer=tokenizer,
        prefix_key=f"{field}.",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    return predict_tools.predict(
        dataset,
        trainer=trainer,
        cache_dir=cache_dir,
        model=model,
        model_output_key=output_key,
        collate_fn=collate_fn,
        loader_kwargs=loader_config,
        validate_store=10_000,
    )
