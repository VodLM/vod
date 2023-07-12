from __future__ import annotations

import pathlib
from typing import Any, Callable, Optional

import elasticsearch
import numpy as np
import omegaconf
import pydantic
from lightning.pytorch import utilities as pl_utils
from loguru import logger
from typing_extensions import Self, Type

from raffle_ds_research.tools import dstruct, index_tools
from raffle_ds_research.tools.index_tools.multi_search import MultiSearchMaster


class SearchConfig(pydantic.BaseModel):
    """Base configuration for search engines (e.g., bm25, faiss)."""

    class Config:
        """Pydantic configuration."""

        extra = pydantic.Extra.forbid

    text_key: str = "text"
    group_key: str = "group_hash"
    faiss: Optional[index_tools.FaissFactoryConfig] = None
    bm25: Optional[index_tools.Bm25FactoryConfig] = None

    @pydantic.validator("faiss", pre=True)
    def _validate_faiss(cls, v: dict | None) -> dict | None:
        if v is None:
            return None

        if isinstance(v, (dict, omegaconf.DictConfig)):
            return index_tools.FaissFactoryConfig(**v).dict()

        return v

    @pydantic.validator("bm25", pre=True)
    def _validate_bm25(cls, v: dict | None) -> dict | None:
        if v is None:
            return None

        if isinstance(v, (dict, omegaconf.DictConfig)):
            return index_tools.Bm25FactoryConfig(**v).dict()

        return v

    @classmethod
    def parse(cls: Type[Self], obj: dict | omegaconf.DictConfig) -> Self:
        """Parse a config object."""
        return cls(**obj)  # type: ignore


def build_search_engine(
    *,
    sections: None | dstruct.SizedDataset[dict[str, Any]],
    vectors: None | dstruct.SizedDataset[np.ndarray],
    config: SearchConfig,
    cache_dir: str | pathlib.Path,
    faiss_enabled: bool = True,
    bm25_enabled: bool = True,
    skip_setup: bool = False,
    barrier_fn: None | Callable[[str], None] = None,
    serve_on_gpu: bool = False,
    close_existing_es_indices: bool = True,
) -> index_tools.MultiSearchMaster:
    """Build a search engine."""
    servers = {}

    if close_existing_es_indices:
        # Close all indices to avoid hitting memory limits
        _close_all_es_indices()

    if faiss_enabled:
        if vectors is None:
            raise ValueError("`vectors` must be provided if `faiss_enabled`")

        faiss_server = index_tools.build_faiss_index(
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

        bm25_server = index_tools.build_bm25_master(
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
