from __future__ import annotations

import os
import pathlib
from typing import Any, Callable, Optional

import faiss
import numpy as np
import omegaconf
import pydantic
import torch
from loguru import logger

from raffle_ds_research.tools import dstruct, pipes
from raffle_ds_research.tools.index_tools import bm25_tools, faiss_tools, search_server
from raffle_ds_research.tools.index_tools.faiss_tools.build_gpu import FaissGpuConfig

FAISS_METRICS = {
    "l2": faiss.METRIC_L2,
    "inner_product": faiss.METRIC_INNER_PRODUCT,
    "l1": faiss.METRIC_L1,
    "linf": faiss.METRIC_Linf,
    "js": faiss.METRIC_JensenShannon,
}

FAISS_METRICS_INV = {v: k for k, v in FAISS_METRICS.items()}


class FaissFactoryConfig(pydantic.BaseModel):
    """Configures the building of a faiss server."""

    factory: str = "Flat"
    nprobe: int = 16
    metric: int = faiss.METRIC_INNER_PRODUCT
    train_size: Optional[int] = None
    logging_level: str = "DEBUG"
    host: str = "http://localhost"
    port: int = 7678
    gpu: Optional[FaissGpuConfig] = None

    @pydantic.validator("metric", pre=True)
    def _validate_metric(cls, v: str | int) -> int:
        if isinstance(v, int):
            return v

        return FAISS_METRICS[v]

    def fingerprint(self) -> str:
        """Return a fingerprint for this config."""
        excludes = {"host", "port", "logging_level"}
        return pipes.fingerprint(self.dict(exclude=excludes))


class Bm25FactoryConfig(pydantic.BaseModel):
    """Configures the building of a bm25 server."""

    text_key: str = "text"
    group_key: Optional[str] = "group_hash"
    host: str = "http://localhost"
    port: int = 9200
    persistent: bool = True
    es_body: Optional[dict] = None

    @pydantic.validator("es_body", pre=True)
    def _validate_es_body(cls, v: dict | None) -> dict | None:
        if isinstance(v, omegaconf.DictConfig):
            v = omegaconf.OmegaConf.to_container(v, resolve=True)  # type: ignore
        return v

    def fingerprint(self) -> str:
        """Return a fingerprint for this config."""
        excludes = {"host", "port", "persistent"}
        return pipes.fingerprint(self.dict(exclude=excludes))


class EnginefactoryConfig(FaissFactoryConfig, Bm25FactoryConfig):
    """General configuration for the search engine."""


def build_search_client(
    index_type: str,
    *,
    sections: None | dstruct.SizedDataset[dict[str, Any]],
    vectors: None | dstruct.SizedDataset[np.ndarray],
    config: dict[str, Any],
    cache_dir: str | pathlib.Path,
    skip_setup: bool = False,
    barrier_fn: None | Callable[[str], None] = None,
    serve_on_gpu: bool = False,
) -> search_server.SearchMaster:
    """Build a search Master client."""
    if sections is not None and vectors is not None and len(sections) != len(vectors):
        raise ValueError(f"Sections and vectors must have the same length. Found: {len(sections)} != {len(vectors)}")
    if index_type == "faiss":
        if vectors is None:
            raise ValueError("Must provide vectors for `faiss` index")
        return build_faiss_index(
            vectors=vectors,
            config=FaissFactoryConfig(**config),
            cache_dir=cache_dir,
            skip_setup=skip_setup,
            barrier_fn=barrier_fn,
            serve_on_gpu=serve_on_gpu,
        )
    if index_type == "bm25":
        if sections is None:
            raise ValueError("Must provide sections for `bm25` index")
        return build_bm25_master(
            sections=sections,
            config=Bm25FactoryConfig(**config),
            skip_setup=skip_setup,
        )

    raise ValueError(f"Unknown index type `{index_type}`")


def build_bm25_master(
    sections: dstruct.SizedDataset[dict[str, Any]],
    config: Bm25FactoryConfig | dict,
    skip_setup: bool = False,
) -> bm25_tools.Bm25Master:
    """Build a bm25 index."""
    if isinstance(config, dict):
        config = Bm25FactoryConfig(**config)
    index_fingerprint = f"{pipes.fingerprint(sections)}-{config.fingerprint()}"
    logger.info(
        f"Init. bm25 index `{index_fingerprint}`, "
        f"text_key: `{config.text_key}`, "
        f"group_key: `{config.group_key}`"
    )
    texts = (row[config.text_key] for row in iter(sections))
    labels = (row[config.group_key] for row in iter(sections)) if config.group_key else None
    return bm25_tools.Bm25Master(
        texts=texts,
        labels=labels,
        host=config.host,
        port=config.port,
        index_name=f"research-{index_fingerprint}",
        input_size=len(sections),
        persistent=config.persistent,
        es_body=config.es_body,
        exist_ok=True,
        skip_setup=skip_setup,
    )


def build_faiss_index(
    vectors: dstruct.SizedDataset[np.ndarray],
    *,
    config: FaissFactoryConfig | dict,
    cache_dir: str | pathlib.Path,
    skip_setup: bool = False,
    barrier_fn: None | Callable[[str], None] = None,
    serve_on_gpu: bool = False,
) -> search_server.SearchMaster:
    """Build a faiss index."""
    if isinstance(config, dict):
        config = FaissFactoryConfig(**config)
    if not torch.cuda.is_available():
        serve_on_gpu = False
    index_fingerprint = f"{pipes.fingerprint(vectors)}-{config.fingerprint()}"
    logger.info(
        f"Init. faiss index `{index_fingerprint}`, "
        f"factory: `{config.factory}`, "
        f"metric: `{FAISS_METRICS_INV[config.metric]}`, "
        f"train_size: `{config.train_size}`"
    )
    index_path = pathlib.Path(cache_dir, "indices", f"{index_fingerprint}.faiss")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if not skip_setup:
        if not index_path.exists():
            logger.info(f"Building faiss index at `{index_path}`")
            index = faiss_tools.build_faiss_index(
                vectors=vectors,
                factory_string=config.factory,
                faiss_metric=config.metric,
                train_size=config.train_size,
                gpu_config=config.gpu,
            )
            logger.info(f"Saving faiss index to `{index_path}`")
            faiss.write_index(index, str(index_path))
            if not index_path.exists():
                raise FileNotFoundError(
                    f"{_rank_info()}Could not find index at `{index_path}` right after building it."
                )
        else:
            logger.info(f"Loading existing faiss index from `{index_path}`")

    if barrier_fn is not None:
        barrier_fn(f"faiss build: `{index_path.name}`")

    if not index_path.exists():
        raise FileNotFoundError(f"{_rank_info()}Could not find faiss index at `{index_path}`.")

    return faiss_tools.FaissMaster(
        index_path=index_path,
        nprobe=config.nprobe,
        logging_level=config.logging_level,
        host=config.host,
        port=config.port,
        skip_setup=skip_setup,
        serve_on_gpu=serve_on_gpu,
    )


def _rank_info() -> str:
    rank = os.getenv("RANK", None)
    winfo = f"[{rank}] " if rank is not None else ""
    return winfo
