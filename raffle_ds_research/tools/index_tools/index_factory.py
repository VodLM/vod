from __future__ import annotations

import pathlib
from typing import Any, Optional

import faiss
import numpy as np
import pydantic
from loguru import logger

from raffle_ds_research.tools import dstruct, pipes
from raffle_ds_research.tools.index_tools import bm25_tools, faiss_tools, search_server

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
    logging_level: str = "CRITICAL"
    host: str = "http://localhost"
    port: int = 7678

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
) -> search_server.SearchMaster:
    """Build a search Master client."""
    if index_type == "faiss":
        if vectors is None:
            raise ValueError("Must provide vectors for `faiss` index")
        return build_faiss_master(
            vectors=vectors,
            config=FaissFactoryConfig(**config),
            cache_dir=cache_dir,
            skip_setup=skip_setup,
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


def build_faiss_master(
    vectors: dstruct.SizedDataset[np.ndarray],
    *,
    config: FaissFactoryConfig,
    cache_dir: str | pathlib.Path,
    skip_setup: bool = False,
) -> search_server.SearchMaster:
    """Build a faiss index."""
    index_fingerprint = f"{pipes.fingerprint(vectors)}-{config.fingerprint()}"
    logger.info(
        f"Init. faiss index `{index_fingerprint}`, "
        f"factory: `{config.factory}`, "
        f"metric: `{FAISS_METRICS_INV[config.metric]}`, "
        f"train_size: `{config.train_size}`"
    )
    index_path = pathlib.Path(cache_dir, f"{index_fingerprint}.faiss")

    if not skip_setup and not index_path.exists():
        logger.info(f"Building faiss index at `{index_path}`")
        index = faiss_tools.build_faiss_master(
            vectors=vectors,
            factory_string=config.factory,
            faiss_metric=config.metric,
            train_size=config.train_size,
        )
        faiss.write_index(index, str(index_path))
    else:
        logger.info(f"Loading existing faiss index from `{index_path}`")

    if not index_path.exists():
        raise FileNotFoundError(f"Could not find index at {index_path}")

    return faiss_tools.FaissMaster(
        index_path=index_path,
        nprobe=config.nprobe,
        logging_level=config.logging_level,
        host=config.host,
        port=config.port,
        skip_setup=skip_setup,
    )


def build_bm25_master(
    sections: dstruct.SizedDataset[dict[str, Any]],
    config: Bm25FactoryConfig,
    skip_setup: bool = False,
) -> bm25_tools.Bm25Master:
    """Build a bm25 index."""
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
        persistent=config.persistent,
        exist_ok=True,
        skip_setup=skip_setup,
    )
