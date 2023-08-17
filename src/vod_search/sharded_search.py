from __future__ import annotations

import asyncio
import collections
import dataclasses
from typing import Any, Generic, Optional, TypeVar

import numpy as np
from vod_search import base, rdtypes

Sc = TypeVar("Sc", bound=base.SearchClient, covariant=True)
Sm = TypeVar("Sm", bound=base.SearchMaster, covariant=True)
K = TypeVar("K", covariant=True)


@dataclasses.dataclass
class _ShardedQueries(Generic[K]):
    shards: dict[K, dict[str, Any]]
    lookup: list[tuple[K, int]]


class ShardedSearchClient(Generic[K, Sc], base.SearchClient):
    """A sharded search client."""

    _shards: dict[K, Sc]
    _offsets: dict[K, int]

    def __init__(self, shards: dict[K, Sc], offsets: dict[K, int]):
        self._shards = shards
        self._offsets = offsets

        if shards.keys() != offsets.keys():
            raise ValueError(
                f"Keys of `shards` and `offsets` must be the same. Found {shards.keys()} and {offsets.keys()}"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shards={self._shards})"

    @property
    def shards(self) -> dict[K, Sc]:
        """The shards."""
        return self._shards.copy()

    @property
    def offsets(self) -> dict[K, int]:
        """The offsets."""
        return self._offsets.copy()

    @property
    def requires_vectors(self) -> bool:
        """Whether the client requires vectors to be passed to the search method."""
        return any(shard.requires_vectors for shard in self.shards.values())

    def ping(self) -> bool:
        """Ping the server."""
        return all(shard.ping() for shard in self.shards.values())

    def search(
        self,
        *,
        text: list[str],
        vector: Optional[np.ndarray] = None,
        group: Optional[list[str | int]] = None,
        section_ids: Optional[list[list[str | int]]] = None,
        shard: Optional[list[K]] = None,
        top_k: int = 3,
    ) -> rdtypes.RetrievalBatch[np.ndarray]:
        """Search the server given a batch of text and/or vectors."""
        if shard is None:
            raise ValueError("Must specify `shard`")
        if set(shard) > set(self.shards.keys()):
            raise ValueError(f"Invalid shard names {shard}. Valid names are {self.shards.keys()}")

        # Scatter queries to the shards
        sharded_queries = _scatter_queries(
            text=text,
            shard=shard,
            vector=vector,
            group=group,
            section_ids=section_ids,
        )

        # Search the shards
        results_by_shard = {}
        for shard_name, query in sharded_queries.shards.items():
            offset = self.offsets[shard_name]
            search_shard = self.shards[shard_name]
            results_by_shard[shard_name] = search_shard.search(
                text=query["text"],
                group=query["group"],
                section_ids=query["section_ids"],
                vector=np.stack(query["vector"]) if vector is not None else None,
                top_k=top_k,
            )
            # Apply the offset
            results_by_shard[shard_name].indices += offset

        # Gather the results and stack them
        return _gather_results(lookup=sharded_queries.lookup, results=results_by_shard)

    async def async_search(
        self,
        *,
        text: list[str],
        shard: Optional[list[K]],
        vector: Optional[np.ndarray] = None,
        group: Optional[list[str | int]] = None,
        section_ids: Optional[list[list[str | int]]] = None,
        top_k: int = 3,
    ) -> rdtypes.RetrievalBatch[np.ndarray]:
        """Search the server given a batch of text and/or vectors."""
        if shard is None:
            raise ValueError("Must specify `shard`")
        if set(shard) > set(self.shards.keys()):
            raise ValueError(f"Invalid shard names {shard}. Valid names are {self.shards.keys()}")

        # Scatter queries to the shards
        sharded_queries = _scatter_queries(
            text=text,
            shard=shard,
            vector=vector,
            group=group,
            section_ids=section_ids,
        )

        # Make the queries
        shard_names: list[K] = list(sharded_queries.shards.keys())
        payloads = [
            {
                "shard_name": shard_name,
                "search_shard": self.shards[shard_name],
                "offset": self.offsets[shard_name],
                "query": sharded_queries.shards[shard_name],
            }
            for shard_name in shard_names
        ]

        def _search_fn(payload: dict[str, Any]) -> rdtypes.RetrievalBatch[np.ndarray]:
            """Search a single shard."""
            result = payload["search_shard"].search(
                text=payload["query"]["text"],
                group=payload["query"]["group"],
                section_ids=payload["query"]["section_ids"],
                vector=np.stack(payload["query"]["vector"]) if vector is not None else None,
                top_k=top_k,
            )
            # Apply the offset
            result.indices += payload["offset"]
            return result

        # Search the shards
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                None,
                _search_fn,
                payload,
            )
            for payload in payloads
        ]

        # Unpack the results
        results_by_shard: dict[K, rdtypes.RetrievalBatch[np.ndarray]] = dict(
            zip(shard_names, await asyncio.gather(*futures))
        )

        # Gather the results and stack them
        return _gather_results(lookup=sharded_queries.lookup, results=results_by_shard)


def _scatter_queries(
    text: list[str],
    shard: list[K],
    vector: Optional[np.ndarray] = None,
    group: Optional[list[str | int]] = None,
    section_ids: Optional[list[list[str | int]]] = None,
) -> _ShardedQueries[K]:
    shards = collections.defaultdict(lambda: collections.defaultdict(list))
    lookup = []
    for i, shard_name in enumerate(shard):
        shards[shard_name]["text"].append(text[i])
        shards[shard_name]["local_rank"].append(i)
        lookup.append((shard_name, len(shards[shard_name]["text"]) - 1))
        if group is not None:
            shards[shard_name]["group"].append(group[i])
        if section_ids is not None:
            shards[shard_name]["section_ids"].append(section_ids[i])
        if vector is not None:
            shards[shard_name]["vector"].append(vector[i])
    return _ShardedQueries(shards=dict(shards), lookup=lookup)


def _gather_results(
    lookup: list[tuple[K, int]],
    results: dict[K, rdtypes.RetrievalBatch[np.ndarray]],
) -> rdtypes.RetrievalBatch[np.ndarray]:
    gathered_results = [results[name][j] for name, j in lookup]
    return rdtypes.stack_samples(gathered_results)


class ShardedSearchMaster(Generic[K, Sm, Sc], base.SearchMaster[ShardedSearchClient]):
    """Handle multiple search servers."""

    shards: dict[K, Sm]

    def __init__(
        self,
        shards: dict[K, Sm],
        offsets: dict[K, int],
        skip_setup: bool = False,
        free_resources: bool = False,
    ):
        """Initialize the search master."""
        super().__init__(skip_setup=skip_setup, free_resources=free_resources)
        self.shards = shards
        self.offsets = offsets
        if shards.keys() != offsets.keys():
            raise ValueError(
                f"Keys of `shards` and `offsets` must be the same. Found {shards.keys()} and {offsets.keys()}"
            )

    def _make_cmd(self) -> list[str]:
        """Make the command to start the server."""
        raise NotImplementedError(f"{type(self).__name__} does not implement `_make_cmd`")

    def __enter__(self) -> ShardedSearchMaster:
        """Start the servers."""
        for shard in self.shards.values():
            shard.__enter__()

        return self

    def __exit__(self, *args, **kwargs) -> None:  # noqa: ANN, ARG
        """Stop the servers."""
        for shard in self.shards.values():
            shard.__exit__(*args, **kwargs)

    def get_client(self) -> ShardedSearchClient[K, Sc]:
        """Get the client for interacting with the Faiss server."""
        return ShardedSearchClient(
            shards={name: shard.get_client() for name, shard in self.shards.items()},
            offsets=self.offsets,
        )

    def _free_resources(self) -> None:
        for shard in self.shards.values():
            shard._free_resources()
