from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
import uuid
import warnings
from typing import Any, Iterable, Optional

import datasets
import elasticsearch as es
import numpy as np
import rich
import rich.progress
from elasticsearch import helpers as es_helpers
from loguru import logger
from vod_search import base, rdtypes

es_logger = logging.getLogger("elastic_transport")
es_logger.setLevel(logging.WARNING)

ROW_IDX_KEY = "__row_idx__"
BODY_KEY: str = "body"
GROUP_KEY: str = "group"
SECTION_ID_KEY: str = "section_id"
TERMS_BOOST = 10_000  # Set this value high enough to make sure we can identify the terms that have been boosted.
EPS = 1e-5


class ElasticsearchClient(base.SearchClient):
    """ElasticSearch client for interacting for spawning an `elasticsearch` server and querying it."""

    requires_vectors: bool = False
    _client: es.Elasticsearch
    _index_name: str

    def __init__(
        self,
        url: str,
        index_name: str,
        supports_groups: bool = False,
    ):
        self.url = url
        self._client = es.Elasticsearch(url)
        self._index_name = index_name
        self.supports_groups = supports_groups

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}[{self.url}]("
            f"requires_vectors={self.requires_vectors}, "
            f"index_name={self._index_name})"
        )

    def ping(self) -> bool:
        """Ping the server."""
        try:
            client = es.Elasticsearch(self.url)
            return client.ping()
        except es.exceptions.ConnectionError:
            return False

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the client state."""
        state = self.__dict__.copy()
        state.pop("_client", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Recreate the client from the state."""
        self.__dict__.update(state)
        self._client = es.Elasticsearch(self.url)

    def __del__(self) -> None:
        """Close the client."""
        with contextlib.suppress(AttributeError):
            self._client.close()

    def search(
        self,
        *,
        text: list[str],
        vector: Optional[rdtypes.Ts] = None,  # noqa: ARG
        group: Optional[list[str | int]] = None,
        section_ids: Optional[list[list[str | int]]] = None,
        top_k: int = 3,
    ) -> rdtypes.RetrievalBatch[np.ndarray]:
        """Search elasticsearch for the batch of text queries using `msearch`. NB: `vector` is not used here."""
        start_time = time.time()
        if self.supports_groups and group is None:
            warnings.warn(f"This `{type(self).__name__}` supports group, but no label is provided.", stacklevel=2)

        if section_ids is not None and group is None:
            raise ValueError("`section_ids` is only supported when group is provided.")

        queries = self._make_queries(
            text,
            top_k=top_k,
            groups=group,
            section_ids=section_ids,
            terms_boost_value=TERMS_BOOST,
        )

        # Search with retries
        responses = self._client.msearch(searches=queries)
        while "error" in responses:
            logger.warning("Error in response: {}", responses["error"])
            time.sleep(0.2)

        # Unpack the responses
        # TODO: efficient implementation

        # compute the max length of the hits
        max_hits = max(len(response["hits"]["hits"]) for response in responses["responses"])

        # process the responses
        indices, scores = [], []
        for response in responses["responses"]:
            # process the indices
            try:
                hits = response["hits"]["hits"]
            except KeyError:
                rich.print(response)
                raise

            # process the indices
            indices_ = (hit["_source"][ROW_IDX_KEY] for hit in hits[:max_hits])
            indices_ = np.fromiter(indices_, dtype=np.int64)
            indices_ = np.pad(indices_, (0, max_hits - len(indices_)), constant_values=-1)

            # process the scores and the labels
            scores_ = (hit["_score"] for hit in hits[:max_hits])
            scores_ = np.fromiter(scores_, dtype=np.float32)
            scores_ = np.pad(scores_, (0, max_hits - len(scores_)), constant_values=-np.inf)

            indices.append(indices_)
            scores.append(scores_)

        # Stack the results
        indices = np.stack(indices)
        scores = np.stack(scores)
        if section_ids is not None:
            # process the labels (super hacky)
            labels = (scores >= TERMS_BOOST).astype(np.int64)
            labels[indices < 0] = -1  # Pad with -1
            scores[labels > 0] -= TERMS_BOOST  # Re-adjust the scores
            scores = np.where(scores < EPS, np.nan, scores)  # set the scores to NaN for documents with score zero
        else:
            labels = None

        return rdtypes.RetrievalBatch(
            indices=indices,
            scores=scores,
            labels=labels,
            meta={"time": time.time() - start_time},
        )

    def _make_queries(
        self,
        texts: list[str],
        *,
        top_k: int,
        groups: Optional[list[str | int]] = None,
        section_ids: Optional[list[list[str | int]]] = None,
        terms_boost_value: float = 1.0,
    ) -> list:
        if groups is None:
            groups = []
        if section_ids is None:
            section_ids = []

        def _make_search_body(
            text: str,
            group: Optional[str | int] = None,
            section_ids: Optional[list[str | int]] = None,
            terms_boost_value: float = 1.0,
        ) -> dict[str, Any]:
            body = {"should": []}
            if len(text):
                body["should"].append(
                    {"match": {BODY_KEY: text}},
                )
            if section_ids is not None:
                body["should"].append(
                    {"terms": {SECTION_ID_KEY: section_ids, "boost": terms_boost_value}},  # type: ignore
                )
            if group is not None:
                body["filter"] = {"term": {GROUP_KEY: group}}  # type: ignore
            return {"query": {"bool": body}}

        return [
            part
            for text, group, ids in itertools.zip_longest(texts, groups, section_ids)
            for part in [
                {"index": self._index_name},
                {
                    "from": 0,
                    "size": top_k,
                    "_source": [ROW_IDX_KEY],
                    **_make_search_body(
                        text,
                        group=group,
                        section_ids=ids,
                        terms_boost_value=terms_boost_value,
                    ),
                },
            ]
        ]


def _yield_input_data(
    texts: Iterable[str],
    groups: Optional[Iterable[str | int]] = None,
    section_ids: Optional[Iterable[str | int]] = None,
) -> Iterable[dict[str, Any]]:
    """Yield the input data for indexing."""
    for row_idx, (text, group, section_id) in enumerate(
        itertools.zip_longest(
            texts,
            groups or [],
            section_ids or [],
        ),
    ):
        yield {
            ROW_IDX_KEY: row_idx,
            BODY_KEY: text,
            GROUP_KEY: group,
            SECTION_ID_KEY: section_id,
        }


class ElasticSearchMaster(base.SearchMaster[ElasticsearchClient]):
    """Handles a BM25 search server."""

    _allow_existing_server: bool = True

    def __init__(
        self,
        texts: Iterable[str],
        *,
        groups: Optional[Iterable[str | int]] = None,
        section_ids: Optional[Iterable[str | int]] = None,
        host: str = "http://localhost",
        port: int = 9200,  # hardcoded for now
        index_name: Optional[str] = None,
        input_size: Optional[int] = None,
        persistent: bool = False,
        exist_ok: bool = False,
        skip_setup: bool = False,
        es_body: Optional[dict] = None,
        **proc_kwargs: Any,
    ):
        super().__init__(skip_setup=skip_setup)
        self._host = host
        self._port = port
        self._proc_kwargs = proc_kwargs
        self._input_texts = texts
        self._input_groups = groups
        self._input_section_ids = section_ids
        self._es_body = es_body
        if index_name is None:
            index_name = f"auto-{uuid.uuid4().hex}"
        self._index_name = index_name
        self._persistent = persistent
        self._exist_ok = exist_ok

        if input_size is None and isinstance(texts, (list, datasets.Dataset)):
            input_size = len(texts)
        self._input_data_size = input_size

    @property
    def supports_groups(self) -> bool:
        """Whether the index supports labels."""
        return self._input_groups is not None

    def _on_init(self) -> None:
        stream = rich.progress.track(
            _yield_input_data(
                texts=self._input_texts,
                groups=self._input_groups,
                section_ids=self._input_section_ids,
            ),
            description=f"Building ES index {self._index_name}",
            total=self._input_data_size,
        )
        maybe_ingest_data(
            stream,
            url=self.url,
            index_name=self._index_name,
            exist_ok=self._exist_ok,
            es_body=self._es_body,
        )

    def _on_exit(self) -> None:
        client = es.Elasticsearch(self.url)
        if not self._persistent:
            client.indices.delete(index=self._index_name)
        else:
            client.indices.close(index=self._index_name)

    def _make_cmd(self) -> list[str]:
        return ["elasticsearch"]

    @property
    def url(self) -> str:
        """Get the url of the search server."""
        return f"{self._host}:{self._port}"

    @property
    def service_info(self) -> str:
        """Return the name of the service."""
        return f"Elasticsearch[{self.url}]"

    def get_client(self) -> ElasticsearchClient:
        """Get a client to the search server."""
        return ElasticsearchClient(self.url, index_name=self._index_name, supports_groups=self.supports_groups)


def maybe_ingest_data(
    stream: Iterable[dict[str, Any]],
    *,
    url: str,
    index_name: str,
    chunk_size: int = 1000,
    exist_ok: bool = False,
    es_body: Optional[dict] = None,
) -> None:
    """Ingest data into Elasticsearch."""
    asyncio.run(
        _async_maybe_ingest_data(
            stream, url=url, index_name=index_name, chunk_size=chunk_size, exist_ok=exist_ok, es_body=es_body
        )
    )


async def _async_maybe_ingest_data(
    stream: Iterable[dict[str, Any]],
    *,
    url: str,
    index_name: str,
    chunk_size: int = 1000,
    exist_ok: bool = False,
    es_body: Optional[dict] = None,
) -> None:
    async with es.AsyncElasticsearch(url) as client:
        if await client.indices.exists(index=index_name):
            if exist_ok:
                logger.info(f"Re-using existing ES index `{index_name}`")
                await client.indices.open(index=index_name)
                return
            raise RuntimeError(f"Index {index_name} already exists")
        try:
            logger.info(f"Creating new ES index `{index_name}`")
            await client.indices.create(index=index_name, body=es_body)
            actions = ({"_index": index_name, "_source": eg} for eg in stream)
            await es_helpers.async_bulk(
                client,
                actions=actions,
                chunk_size=chunk_size,
                refresh=True,
            )

        except Exception as e:
            await client.indices.delete(index=index_name)
            raise e
