
from __future__ import annotations

import abc
import asyncio
import contextlib
import itertools
import logging
import subprocess
import time
import uuid
import warnings
from typing import Any, Iterable, Optional

import datasets
import elasticsearch as es
import loguru
import numpy as np
import rich
import rich.progress
from elasticsearch import helpers as es_helpers
from loguru import logger
from typing_extensions import Self
from vod_search import rdtypes as rtypes
from vod_search import search_server


class QdrantSearchClient(search_server.SearchClient):
    """A client to interact with a search server."""

    requires_vectors: bool = True

    @abc.abstractmethod
    def ping(self) -> bool:
        """Ping the server."""
        raise NotImplementedError()

    @abc.abstractmethod
    def search(
        self,
        *,
        text: list[str],
        vector: Optional[rtypes.Ts] = None,
        group: Optional[list[str | int]] = None,
        section_ids: Optional[list[list[str | int]]] = None,
        top_k: int = 3,
    ) -> rtypes.RetrievalBatch[rtypes.Ts]:
        """Search the server given a batch of text and/or vectors."""
        raise NotImplementedError()

    async def async_search(
        self,
        *,
        text: list[str],
        vector: Optional[rtypes.Ts] = None,
        group: Optional[list[str | int]] = None,
        section_ids: Optional[list[list[str | int]]] = None,
        top_k: int = 3,
    ) -> rtypes.RetrievalBatch[rtypes.Ts]:
        """Search the server given a batch of text and/or vectors."""
        return self.search(
            text=text,
            vector=vector,
            group=group,
            section_ids=section_ids,
            top_k=top_k,
        )



class QdrantSearchMaster(search_server.SearchMaster[QdrantSearchClient], abc.ABC):
    """A class that manages a search server."""

    def __init__(self, skip_setup: bool = False) -> None:
        self.skip_setup = skip_setup

    def __enter__(self) -> Self:
        """Start the server."""
        if not self.skip_setup:
            self._setup()
        return self

    def _on_init(self) -> None:
        pass

    def _on_exit(self) -> None:
        pass

    @abc.abstractmethod
    def get_client(self) -> QdrantSearchClient:
        """Return a client to the server."""
        raise NotImplementedError

    @abc.abstractmethod
    def _make_cmd(self) -> list[str]:
        raise NotImplementedError