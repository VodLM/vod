import asyncio
import collections
import contextlib
import itertools
import logging
import time
import typing as typ
import uuid
import warnings

import datasets
import elasticsearch as es
import numpy as np
import rich
import rich.progress
import vod_types as vt
from elasticsearch import helpers as es_helpers
from loguru import logger
from vod_configs.es_body import (
    BODY_KEY,
    ROW_IDX_KEY,
    SECTION_ID_KEY,
    SUBSET_ID_KEY,
    validate_es_body,
)
from vod_search import base

es_logger = logging.getLogger("elastic_transport")
es_logger.setLevel(logging.WARNING)


class ElasticsearchClient(base.SearchClient):
    """ElasticSearch client for interacting for spawning an `elasticsearch` server and querying it."""

    requires_vectors: bool = False
    _client: es.Elasticsearch
    _index_name: str

    def __init__(
        self,
        url: str,
        index_name: str,
        supports_subsets: bool = False,
    ):
        self.url = url
        self._client = es.Elasticsearch(url)
        self._index_name = index_name
        self.support_subsets = supports_subsets

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

    def __getstate__(self) -> dict[str, typ.Any]:
        """Serialize the client state."""
        state = self.__dict__.copy()
        state.pop("_client", None)
        return state

    def __setstate__(self, state: dict[str, typ.Any]) -> None:
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
        vector: None | np.ndarray = None,  # noqa: ARG002
        subset_ids: None | list[list[base.SubsetId]] = None,
        ids: None | list[list[base.SectionId]] = None,
        shard: None | list[base.ShardName] = None,  # noqa: ARG002
        top_k: int = 3,
    ) -> vt.RetrievalBatch:
        """Search elasticsearch for the batch of text queries using `msearch`. NOTE: `vector` is not used here."""
        start_time = time.time()
        if self.support_subsets and subset_ids is None:
            warnings.warn(f"This `{type(self).__name__}` supports subset ids, but no label is provided.", stacklevel=2)

        queries = self._make_queries(
            text,
            top_k=top_k,
            subset_ids=subset_ids,
            ids=ids,
        )

        # Search with retries
        responses = self._client.msearch(searches=queries)
        while "error" in responses:
            logger.warning("Error in response: {}", responses["error"])
            time.sleep(0.2)

        # Unpack the responses
        # TODO: efficient implementation

        # compute the max length of the hits
        try:
            max_hits = max(len(response["hits"]["hits"]) for response in responses["responses"])
        except KeyError:
            rich.print(responses)
            raise

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
        labels = (scores > -np.inf).astype(np.int64) if ids is not None else None

        return vt.RetrievalBatch(
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
        subset_ids: None | list[base.SubsetId] | list[list[base.SubsetId]] = None,
        ids: None | list[base.SectionId] | list[list[base.SectionId]] = None,
    ) -> list:
        if subset_ids is None:
            subset_ids = []
        if ids is None:
            ids = []

        def _make_search_body(
            text: str,
            subset_ids: None | str | list[str] = None,
            section_ids: None | str | list[str] = None,
        ) -> dict[str, typ.Any]:
            body = collections.defaultdict(list)
            if len(text):
                body["should"].append(
                    {"match": {BODY_KEY: text}},
                )
            if section_ids is not None:
                # Still search the index for `[]`, this will return zero results
                # This is necessary to implement the correct behaviour when
                # looking up positive sections with no label.
                if isinstance(section_ids, str):
                    section_ids = [section_ids]
                body["filter"].append({"terms": {SECTION_ID_KEY: section_ids}})

            if subset_ids is not None and subset_ids != []:
                # We don't include `subset_ids` in the query
                # when it's an empty list. This means when no `subset_ids` are provided
                # we don't filter by `subset_ids`.
                if isinstance(subset_ids, str):
                    subset_ids = [subset_ids]
                body["filter"].append({"terms": {SUBSET_ID_KEY: subset_ids}})

            return {"query": {"bool": dict(body)}}

        return [
            part
            for text, sub_ids, sec_ids in itertools.zip_longest(texts, subset_ids, ids)
            for part in [
                {"index": self._index_name},
                {
                    "from": 0,
                    "size": top_k,
                    "_source": [
                        ROW_IDX_KEY,
                    ],
                    **_make_search_body(
                        text,
                        subset_ids=sub_ids,
                        section_ids=sec_ids,
                    ),
                },
            ]
        ]


def _yield_input_data(
    texts: typ.Iterable[str],
    subset_ids: None | typ.Iterable[base.SubsetId] = None,
    section_ids: None | typ.Iterable[base.SectionId] = None,
) -> typ.Iterable[dict[str, int | str | base.SubsetId | base.SectionId]]:
    """Yield the input data for indexing."""
    for row_idx, (text, subset_id, section_id) in enumerate(
        itertools.zip_longest(
            texts,
            subset_ids or [],
            section_ids or [],
        ),
    ):
        yield {
            ROW_IDX_KEY: row_idx,
            BODY_KEY: text,
            SUBSET_ID_KEY: subset_id,
            SECTION_ID_KEY: section_id,
        }


class ElasticSearchMaster(base.SearchMaster[ElasticsearchClient]):
    """Handles a BM25 search server."""

    _allow_existing_server: bool = True

    def __init__(
        self,
        texts: typ.Iterable[str],
        *,
        subset_ids: None | typ.Iterable[base.SubsetId] = None,
        section_ids: None | typ.Iterable[base.SectionId] = None,
        host: str = "http://localhost",
        port: int = 9200,  # hardcoded for now
        index_name: None | str = None,
        input_size: None | int = None,
        persistent: bool = False,
        exist_ok: bool = False,
        skip_setup: bool = False,
        free_resources: bool = False,
        es_body: None | dict[str, typ.Any] = None,
        language: None | str = None,
        **proc_kwargs: typ.Any,
    ):
        super().__init__(skip_setup=skip_setup, free_resources=free_resources)
        self._host = host
        self._port = port
        self._proc_kwargs = proc_kwargs
        self._input_texts = texts
        self._input_subset_ids = subset_ids
        self._input_section_ids = section_ids
        self._es_body = validate_es_body(es_body, language=language)
        if index_name is None:
            index_name = f"auto-{uuid.uuid4().hex}"
        self._index_name = index_name
        self._persistent = persistent
        self._exist_ok = exist_ok

        if input_size is None and isinstance(texts, (list, datasets.Dataset)):
            input_size = len(texts)
        self._input_data_size = input_size

    @property
    def support_subset_ids(self) -> bool:
        """Whether the index supports labels."""
        return self._input_subset_ids is not None

    def _on_init(self) -> None:
        stream = rich.progress.track(
            _yield_input_data(
                texts=self._input_texts,
                subset_ids=self._input_subset_ids,
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
        return ElasticsearchClient(self.url, index_name=self._index_name, supports_subsets=self.support_subset_ids)

    def _free_resources(self) -> None:
        _close_all_es_indices(self.url)


def maybe_ingest_data(
    stream: typ.Iterable[dict[str, typ.Any]],
    *,
    url: str,
    index_name: str,
    chunk_size: int = 1000,
    exist_ok: bool = False,
    es_body: None | dict[str, typ.Any] = None,
) -> None:
    """Ingest data into Elasticsearch."""
    asyncio.run(
        _async_maybe_ingest_data(
            stream, url=url, index_name=index_name, chunk_size=chunk_size, exist_ok=exist_ok, es_body=es_body
        )
    )


async def _async_maybe_ingest_data(
    stream: typ.Iterable[dict[str, typ.Any]],
    *,
    url: str,
    index_name: str,
    chunk_size: int = 1000,
    exist_ok: bool = False,
    es_body: None | dict[str, typ.Any] = None,
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


def _close_all_es_indices(es_url: str = "http://localhost:9200") -> None:
    """Close all `elasticsearch` indices."""
    try:
        client = es.Elasticsearch(es_url)
        for index_name in client.indices.get(index="*"):
            if index_name.startswith("."):
                continue
            try:
                if client.indices.exists(index=index_name):
                    logger.debug(f"Elasticsearch: closing index `{es_url}/{index_name}`")
                    client.indices.close(index=index_name)
            except Exception as exc:
                logger.warning(f"Could not close index `{index_name}`: {exc}")
    except Exception as exc:
        logger.warning(f"Could not connect to ES: {exc}")
