from __future__ import annotations

import pathlib
from typing import Any, Callable

import numpy as np
import omegaconf
import pydantic
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
    faiss: index_tools.FaissFactoryConfig
    bm25: index_tools.Bm25FactoryConfig

    @classmethod
    def parse(cls: Type[Self], obj: dict | omegaconf.DictConfig) -> Self:
        """Parse a config object."""
        return cls(**obj)  # type: ignore


def build_search_engine(
    *,
    sections: None | dstruct.SizedDataset[dict[str, Any]],
    vectors: None | dstruct.SizedDataset[np.ndarray],
    config: SearchConfig,
    cache_dir: pathlib.Path,
    faiss_enabled: bool = True,
    bm25_enabled: bool = True,
    skip_setup: bool = False,
    barrier_fn: None | Callable[[str], None] = None,
) -> index_tools.MultiSearchMaster:
    """Build a search engine."""
    servers = {}

    if faiss_enabled:
        if vectors is None:
            raise ValueError("`vectors` must be provided if `faiss_enabled`")

        faiss_server = index_tools.build_faiss_index(
            vectors=vectors,
            config=config.faiss,
            cache_dir=cache_dir,
            skip_setup=skip_setup,
            barrier_fn=barrier_fn,
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
