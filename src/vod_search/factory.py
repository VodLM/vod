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
from vod_search import base, es_search, faiss_search, qdrant_search
from vod_search.multi_search import MultiSearchMaster
from vod_tools import dstruct, pipes

from src import vod_configs


def build_search_index(
    index_type: str,
    *,
    sections: None | dstruct.SizedDataset[dict[str, Any]],
    vectors: None | dstruct.SizedDataset[np.ndarray],
    config: dict[str, Any],
    cache_dir: str | pathlib.Path,
    skip_setup: bool = False,
    barrier_fn: None | Callable[[str], None] = None,
    serve_on_gpu: bool = False,
) -> base.SearchMaster:
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
    if index_type == "elasticsearch":
        if sections is None:
            raise ValueError("Must provide sections for `elasticsearch` index")
        return build_elasticsearch_index(
            sections=sections,
            config=vod_configs.ElasticsearchFactoryConfig(**config),
            skip_setup=skip_setup,
        )

    if index_type == "qdrant":
        if vectors is None:
            raise ValueError("Must provide vectors for `qdrant` index")
        if sections is None:
            raise ValueError("Must provide sections for `qdrant` index")
        return build_qdrant_index(
            vectors=vectors,
            sections=sections,
            config=vod_configs.QdrantFactoryConfig(**config),
            skip_setup=skip_setup,
        )

    raise ValueError(f"Unknown index type `{index_type}`")


def build_elasticsearch_index(
    sections: dstruct.SizedDataset[dict[str, Any]],
    config: vod_configs.ElasticsearchFactoryConfig | dict,
    skip_setup: bool = False,
) -> es_search.ElasticSearchMaster:
    """Build a sparse elasticsearch index."""
    if isinstance(config, dict):
        config = vod_configs.ElasticsearchFactoryConfig(**config)
    index_fingerprint = f"{pipes.fingerprint(sections)}-{config.fingerprint()}"
    logger.info(
        f"Init. elasticsearch index `{index_fingerprint}`, "
        f"text_key: `{config.text_key}`, "
        f"group_key: `{config.group_key}`"
    )
    texts = (row[config.text_key] for row in iter(sections))
    groups = (row[config.group_key] for row in iter(sections)) if config.group_key else None
    sections_ids = (row[config.section_id_key] for row in iter(sections)) if config.section_id_key else None
    return es_search.ElasticSearchMaster(
        texts=texts,
        groups=groups,
        section_ids=sections_ids,
        host=config.host,
        port=config.port,
        index_name=f"vod-{index_fingerprint}",
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
) -> faiss_search.FaissMaster:
    """Build a faiss index."""
    if isinstance(config, dict):
        config = vod_configs.FaissFactoryConfig(**config)
    if not torch.cuda.is_available():
        serve_on_gpu = False
    index_fingerprint = f"{pipes.fingerprint(vectors)}-{config.fingerprint()}"
    logger.info(
        f"Init. faiss index `{index_fingerprint}`, "
        f"factory: `{config.factory}`, "
        f"metric: `{vod_configs.FAISS_METRICS_INV[config.metric]}`, "
        f"train_size: `{config.train_size}`"
    )
    index_path = pathlib.Path(cache_dir, "indices", f"{index_fingerprint}.faiss")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if not skip_setup:
        if not index_path.exists():
            logger.info(f"Building faiss index at `{index_path}`")
            index = faiss_search.build_faiss_index(
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

    return faiss_search.FaissMaster(
        index_path=index_path,
        nprobe=config.nprobe,
        logging_level=config.logging_level,
        host=config.host,
        port=config.port,
        skip_setup=skip_setup,
        serve_on_gpu=serve_on_gpu,
    )


def build_qdrant_index(
    vectors: dstruct.SizedDataset[np.ndarray],
    sections: dstruct.SizedDataset[dict[str, Any]],
    config: vod_configs.QdrantFactoryConfig | dict,
    skip_setup: bool = False,
) -> qdrant_search.QdrantSearchMaster:
    """Build a dense Qdrant index."""
    if isinstance(config, dict):
        config = vod_configs.QdrantFactoryConfig(**config)
    index_fingerprint = f"{pipes.fingerprint(vectors)}-{config.fingerprint()}"
    logger.info(
        f"Init. Qdrant index `{index_fingerprint}`, "
        f"vector: `{[len(vectors), *vectors[0].shape]}`, "
        f"group_key: `{config.group_key}`"
    )
    groups = (row[config.group_key] for row in iter(sections)) if config.group_key else None
    return qdrant_search.QdrantSearchMaster(
        vectors=vectors,
        groups=groups,
        host=config.host,
        port=config.port,
        grpc_port=config.grpc_port,
        index_name=f"vod-{index_fingerprint}",
        persistent=config.persistent,
        qdrant_body=config.qdrant_body,
        search_params=config.search_params,
        exist_ok=True,
        skip_setup=skip_setup,
        force_single_collection=config.force_single_collection,
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
    dense_enabled: bool = True,
    sparse_enabled: bool = True,
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

    if dense_enabled:
        if vectors is None:
            raise ValueError("`vectors` must be provided if `dense_enabled`")
        if config.dense is None:
            raise ValueError("`dense` must be provided if `dense_enabled`")

        servers["dense"] = _init_dense_search_engine(
            sections=sections,
            vectors=vectors,
            config=config.dense,
            cache_dir=cache_dir,
            skip_setup=skip_setup,
            barrier_fn=barrier_fn,
            serve_on_gpu=serve_on_gpu,
        )

    if sparse_enabled:
        if sections is None:
            raise ValueError("`sections` must be provided if `sparse_enabled`")
        if config.sparse is None:
            raise ValueError("`sparse` must be provided if `sparse_enabled`")

        sparse_server = build_elasticsearch_index(
            sections=sections,
            config=config.sparse,
            skip_setup=skip_setup,
        )
        servers["sparse"] = sparse_server

    if len(servers) == 0:
        raise ValueError("No search servers were enabled.")

    return MultiSearchMaster(servers=servers, skip_setup=skip_setup)


def _init_dense_search_engine(
    sections: None | dstruct.SizedDataset[dict[str, Any]],
    vectors: dstruct.SizedDataset[np.ndarray],
    config: vod_configs.FaissFactoryConfig | vod_configs.QdrantFactoryConfig,
    cache_dir: str | pathlib.Path,
    skip_setup: bool,
    barrier_fn: None | Callable[[str], None],
    serve_on_gpu: bool,
) -> qdrant_search.QdrantSearchMaster | faiss_search.FaissMaster:
    if isinstance(config, vod_configs.FaissFactoryConfig):
        return build_faiss_index(
            vectors=vectors,
            config=config,
            cache_dir=cache_dir,
            skip_setup=skip_setup,
            barrier_fn=barrier_fn,
            serve_on_gpu=serve_on_gpu,
        )
    if isinstance(config, vod_configs.QdrantFactoryConfig):
        if sections is None:
            raise ValueError("`sections` must be provided when using `Qdrant` search engine")
        return build_qdrant_index(
            vectors=vectors,
            sections=sections,
            config=config,
            skip_setup=skip_setup,
        )

    raise ValueError(f"Unknown dense factory config type `{type(config)}`")


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
