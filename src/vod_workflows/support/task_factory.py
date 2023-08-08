from __future__ import annotations

import dataclasses
from typing import Generic, Optional

import datasets
import typing_extensions
from vod_tools import dstruct

from src import vod_configs


@dataclasses.dataclass
class DsetWithVectors:
    """A dataset with its embeddings."""

    dset: datasets.Dataset
    vectors: Optional[dstruct.TensorStoreFactory]


K = typing_extensions.TypeVar("K")


@dataclasses.dataclass
class RetrievalTask(Generic[K]):
    """A retrieval task with queries, sections and the config required to build a search engine."""

    queries: dict[K, DsetWithVectors]
    sections: dict[K, DsetWithVectors]
    search: vod_configs.MutliSearchFactoryConfig
