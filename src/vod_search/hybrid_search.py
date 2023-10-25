import asyncio
import copy
import typing as typ

import numpy as np
import vod_types as vt
from typing_extensions import Self

from .base import (
    SearchClient,
    SearchMaster,
    SectionId,
    ShardName,
    SubsetId,
)

ClientName: typ.TypeAlias = str


class HybridSearchClient(SearchClient):
    """A client to interact with a search server."""

    clients: dict[ClientName, SearchClient]
    _shard_list: None | list[ShardName]
    _sections: None | vt.DictsSequence

    def __init__(
        self,
        clients: dict[ClientName, SearchClient],
        shard_list: None | list[ShardName] = None,
        sections: None | vt.DictsSequence = None,
    ) -> None:
        self.clients = clients
        self._shard_list = shard_list
        self._sections = sections

    @property
    def shard_list(self) -> list[ShardName]:
        """Get the available shards."""
        if self._shard_list is None:
            raise ValueError("The shard list has not been set.")
        return copy.copy(self._shard_list)

    @property
    def sections(self) -> vt.DictsSequence:
        """Get the sections."""
        if self._sections is None:
            raise ValueError("The sections have not been set.")
        return self._sections

    def __repr__(self) -> str:
        return f"{type(self).__name__}(clients={self.clients}, shards={self.shard_list})"

    @property
    def requires_vectors(self) -> bool:
        """Whether the client requires vectors to be passed to the search method."""
        return any(client.requires_vectors for client in self.clients.values())

    def ping(self) -> bool:
        """Ping the server."""
        return all(client.ping() for client in self.clients.values())

    def search(
        self,
        *,
        text: list[str],
        vector: None | np.ndarray = None,
        subset_ids: None | list[list[SubsetId]] = None,
        ids: None | list[list[SectionId]] = None,
        shard: None | list[ShardName] = None,
        top_k: int = 3,
    ) -> dict[ClientName, vt.RetrievalBatch]:
        """Search the server given a batch of text and/or vectors."""
        return {
            name: client.search(
                vector=vector,
                text=text,
                subset_ids=subset_ids,
                ids=ids,
                shard=shard,
                top_k=top_k,
            )
            for name, client in self.clients.items()
        }

    async def async_search(
        self,
        *,
        text: list[str],
        vector: None | np.ndarray = None,
        subset_ids: None | list[list[SubsetId]] = None,
        ids: None | list[list[SectionId]] = None,
        shard: None | list[ShardName] = None,
        top_k: int = 3,
    ) -> dict[ClientName, vt.RetrievalBatch]:
        """Search the server given a batch of text and/or vectors."""

        def search_fn(args: dict[str, typ.Any]) -> vt.RetrievalBatch:
            client = args.pop("client")
            return client.search(**args)

        names = list(self.clients.keys())
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                None,
                search_fn,
                {
                    "client": self.clients[name],
                    "vector": vector,
                    "text": text,
                    "ids": ids,
                    "subset_ids": subset_ids,
                    "shard": shard,
                    "top_k": top_k,
                },
            )
            for name in names
        ]

        results = await asyncio.gather(*futures)
        return dict(zip(names, results))


class HyrbidSearchMaster(SearchMaster):
    """Handle multiple search servers."""

    servers: dict[str, SearchMaster]
    _shard_list: None | list[ShardName]
    _sections: None | vt.DictsSequence

    def __init__(
        self,
        servers: dict[str, SearchMaster],
        skip_setup: bool = False,
        free_resources: bool = False,
        shard_list: None | list[ShardName] = None,
        sections: None | vt.DictsSequence = None,
    ):
        """Initialize the search master."""
        self.skip_setup = skip_setup
        self.servers = servers
        self.free_resources = free_resources
        self._shard_list = shard_list
        self._sections = sections

    @property
    def shard_list(self) -> list[ShardName]:
        """Get the available shards."""
        if self._shard_list is None:
            raise ValueError("The shard list has not been set.")
        return copy.copy(self._shard_list)

    @property
    def sections(self) -> vt.DictsSequence:
        """Get the sections."""
        if self._sections is None:
            raise ValueError("The sections have not been set.")
        return self._sections

    def __enter__(self) -> Self:
        """Start the servers."""
        if self.free_resources:
            self._free_resources()

        for server in self.servers.values():
            server.__enter__()

        return self

    def _make_cmd(self) -> list[str]:
        raise NotImplementedError(f"{type(self).__name__} does not implement `_make_cmd`")

    def __exit__(self, *args, **kwargs) -> None:  # noqa: ANN003, ANN002
        """Stop the servers."""
        for server in self.servers.values():
            server.__exit__(*args, **kwargs)

    def get_client(self) -> HybridSearchClient:
        """Get the client for interacting with the Faiss server."""
        return HybridSearchClient(
            clients={name: server.get_client() for name, server in self.servers.items()},
            shard_list=self._shard_list,
            sections=self._sections,
        )

    def _free_resources(self) -> None:
        for server in self.servers.values():
            server._free_resources()
