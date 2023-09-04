from __future__ import annotations

import asyncio
from typing import Any, Optional

import numpy as np
from vod_search import base, rdtypes


class HybridSearchClient(base.SearchClient):
    """A client to interact with a search server."""

    clients: dict[str, base.SearchClient]

    def __init__(self, clients: dict[str, base.SearchClient]) -> None:
        self.clients = clients

    def __repr__(self) -> str:
        return f"{type(self).__name__}(clients={self.clients})"

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
        vector: Optional[np.ndarray] = None,
        group: Optional[list[str | int]] = None,
        section_ids: Optional[list[list[str | int]]] = None,
        shard: Optional[list[str]] = None,
        top_k: int = 3,
    ) -> dict[str, rdtypes.RetrievalBatch[np.ndarray]]:
        """Search the server given a batch of text and/or vectors."""
        return {
            name: client.search(
                vector=vector,
                text=text,
                subset_ids=group,
                section_ids=section_ids,
                shard=shard,
                top_k=top_k,
            )
            for name, client in self.clients.items()
        }

    async def async_search(
        self,
        *,
        text: list[str],
        vector: Optional[rdtypes.Ts] = None,
        group: Optional[list[str | int]] = None,
        shard: Optional[list[str]] = None,
        top_k: int = 3,
    ) -> dict[str, rdtypes.RetrievalBatch[np.ndarray]]:
        """Search the server given a batch of text and/or vectors."""

        def search_fn(args: dict[str, Any]) -> rdtypes.RetrievalBatch[np.ndarray]:
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
                    "group": group,
                    "shard": shard,
                    "top_k": top_k,
                },
            )
            for name in names
        ]

        results = await asyncio.gather(*futures)
        return dict(zip(names, results))


class HyrbidSearchMaster(base.SearchMaster):
    """Handle multiple search servers."""

    servers: dict[str, base.SearchMaster]

    def __init__(
        self,
        servers: dict[str, base.SearchMaster],
        skip_setup: bool = False,
        free_resources: bool = False,
    ):
        """Initialize the search master."""
        self.skip_setup = skip_setup
        self.servers = servers
        self.free_resources = free_resources

    def __enter__(self) -> HyrbidSearchMaster:
        """Start the servers."""
        if self.free_resources:
            self._free_resources()

        for server in self.servers.values():
            server.__enter__()

        return self

    def _make_cmd(self) -> list[str]:
        raise NotImplementedError(f"{type(self).__name__} does not implement `_make_cmd`")

    def __exit__(self, *args, **kwargs) -> None:  # noqa: ANN, ARG
        """Stop the servers."""
        for server in self.servers.values():
            server.__exit__(*args, **kwargs)

    def get_client(self) -> HybridSearchClient:
        """Get the client for interacting with the Faiss server."""
        return HybridSearchClient(clients={name: server.get_client() for name, server in self.servers.items()})

    def _free_resources(self) -> None:
        for server in self.servers.values():
            server._free_resources()
