from __future__ import annotations

import os
import pathlib
from typing import Any, Callable

import elasticsearch
import faiss
import numpy as np
import torch
from lightning.pytorch import utilities as pl_utils
from loguru import logger

from src import vod_configs
from src.vod_configs.py.search import FAISS_METRICS_INV
from src.vod_search import bm25_tools, faiss_tools, search_server
from src.vod_search.multi_search import MultiSearchMaster
from src.vod_tools import dstruct, pipes


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
            config=vod_configs.FaissFactoryConfig(**config),
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
            config=vod_configs.Bm25FactoryConfig(**config),
            skip_setup=skip_setup,
        )

    raise ValueError(f"Unknown index type `{index_type}`")


def build_bm25_master(
    sections: dstruct.SizedDataset[dict[str, Any]],
    config: vod_configs.Bm25FactoryConfig | dict,
    skip_setup: bool = False,
) -> bm25_tools.Bm25Master:
    """Build a bm25 index."""
    if isinstance(config, dict):
        config = vod_configs.Bm25FactoryConfig(**config)
    index_fingerprint = f"{pipes.fingerprint(sections)}-{config.fingerprint()}"
    logger.info(
        f"Init. bm25 index `{index_fingerprint}`, "
        f"text_key: `{config.text_key}`, "
        f"group_key: `{config.group_key}`"
    )
    texts = (row[config.text_key] for row in iter(sections))
    groups = (row[config.group_key] for row in iter(sections)) if config.group_key else None
    sections_ids = (row[config.section_id_key] for row in iter(sections)) if config.section_id_key else None
    return bm25_tools.Bm25Master(
        texts=texts,
        groups=groups,
        section_ids=sections_ids,
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
    config: vod_configs.FaissFactoryConfig | dict,
    cache_dir: str | pathlib.Path,
    skip_setup: bool = False,
    barrier_fn: None | Callable[[str], None] = None,
    serve_on_gpu: bool = False,
) -> search_server.SearchMaster:
    """Build a faiss index."""
    if isinstance(config, dict):
        config = vod_configs.FaissFactoryConfig(**config)
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


def build_multi_search_engine(
    *,
    sections: None | dstruct.SizedDataset[dict[str, Any]],
    vectors: None | dstruct.SizedDataset[np.ndarray],
    config: vod_configs.SearchConfig,
    cache_dir: str | pathlib.Path,
    faiss_enabled: bool = True,
    bm25_enabled: bool = True,
    skip_setup: bool = False,
    barrier_fn: None | Callable[[str], None] = None,
    serve_on_gpu: bool = False,
    close_existing_es_indices: bool = True,
) -> MultiSearchMaster:
    """Build a hybrid search engine."""
    servers = {}

    if close_existing_es_indices:
        # Close all indices to avoid hitting memory limits
        _close_all_es_indices()

    if faiss_enabled:
        if vectors is None:
            raise ValueError("`vectors` must be provided if `faiss_enabled`")

        faiss_server = build_faiss_index(
            vectors=vectors,
            config=config.faiss,
            cache_dir=cache_dir,
            skip_setup=skip_setup,
            barrier_fn=barrier_fn,
            serve_on_gpu=serve_on_gpu,
        )
        servers["faiss"] = faiss_server

    if bm25_enabled:
        if sections is None:
            raise ValueError("`sections` must be provided if `bm25_enabled`")

        bm25_server = build_bm25_master(
            sections=sections,
            config=config.bm25,
            skip_setup=skip_setup,
        )
        servers["bm25"] = bm25_server

    if len(servers) == 0:
        raise ValueError("No search servers were enabled.")

    return MultiSearchMaster(servers=servers, skip_setup=skip_setup)


@pl_utils.rank_zero_only
def _close_all_es_indices(es_url: str = "http://localhost:9200") -> None:
    """Close all `elasticsearch` indices."""
    logger.warning(f"Closing all ES indices at `{es_url}`")
    try:
        es = elasticsearch.Elasticsearch(es_url)
        for index_name in es.indices.get(index="*"):
            if index_name.startswith("."):
                continue
            logger.debug(f"Found ES index `{index_name}`")
            try:
                if es.indices.exists(index=index_name):
                    logger.info(f"Closing ES index {index_name}")
                    es.indices.close(index=index_name)
            except Exception as exc:
                logger.warning(f"Could not close index {index_name}: {exc}")
    except Exception as exc:
        logger.warning(f"Could not connect to ES: {exc}")
