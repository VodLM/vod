import asyncio
import logging
import uuid
from typing import Any, Iterable, Optional

import datasets
import elasticsearch as es
import numpy as np
import rich.progress
from elasticsearch import helpers as es_helpers

from raffle_ds_research.tools import predict_tools
from raffle_ds_research.tools.index_tools import retrieval_data_type as rtypes
from raffle_ds_research.tools.index_tools import search_server

es_logger = logging.getLogger("elastic_transport")
es_logger.setLevel(logging.WARNING)

IDX_KEY = "__idx__"
BODY_KEY: str = "body"


class Bm25Client(search_server.SearchClient):
    requires_vectors: bool = False
    _client: es.Elasticsearch
    _index_name: str

    def __init__(self, url: str, index_name: str):
        self.url = url
        self._client = es.Elasticsearch(url)
        self._index_name = index_name

    def ping(self) -> bool:
        try:
            client = es.Elasticsearch(self.url)
            return client.ping()
        except es.exceptions.ConnectionError:
            return False

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_client", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client = es.Elasticsearch(self.url)

    def __del__(self):
        try:
            self._client.close()
        except AttributeError:
            pass

    def search(
        self,
        *,
        text: list[str],
        vector: Optional[rtypes.Ts] = None,
        top_k: int = 3,
    ) -> rtypes.RetrievalBatch[rtypes.Ts]:
        """Search elasticsearch for the batch of text queries using `msearch`.
        `vector` is not used here."""
        queries = self._make_queries(text, top_k)
        responses = self._client.msearch(searches=queries)
        indices, scores = [], []
        for response in responses["responses"]:
            # process the indices
            hits = response["hits"]["hits"]
            indices_ = (hit["_source"][IDX_KEY] for hit in hits[:top_k])
            indices_ = np.fromiter(indices_, dtype=np.int64)
            indices_ = np.pad(indices_, (0, top_k - len(indices_)), constant_values=-1)

            # process the scores
            indices.append(indices_)
            scores_ = (hit["_score"] for hit in hits[:top_k])
            scores_ = np.fromiter(scores_, dtype=np.float32)
            scores_ = np.pad(scores_, (0, top_k - len(scores_)), constant_values=-1)
            scores.append(scores_)

        return rtypes.RetrievalBatch(
            indices=np.stack(indices),
            scores=np.stack(scores),
        )

    def _make_queries(self, text, top_k):
        queries = [
            part
            for text in text
            for part in [
                {"index": self._index_name},
                {
                    "from": 0,
                    "size": top_k,
                    "fields": [IDX_KEY],
                    "query": {"match": {BODY_KEY: text}},
                },
            ]
        ]
        return queries


class Bm25Master(search_server.SearchMaster[Bm25Client]):
    _allow_existing_server: bool = True

    def __init__(
        self,
        input_data: Iterable[str],
        *,
        host: str = "http://localhost",
        port: int = 9200,  # hardcoded for now
        index_name: Optional[str] = None,
        input_size: Optional[int] = None,
        persistent: bool = False,
        exist_ok: bool = False,
        **proc_kwargs: Any,
    ):
        self.host = host
        self.port = port
        self.proc_kwargs = proc_kwargs
        self._input_data = input_data
        if index_name is None:
            index_name = f"auto-{uuid.uuid4().hex}"
        self._index_name = index_name
        self._persistent = persistent
        self._exist_ok = exist_ok

        if input_size is None:
            if isinstance(input_data, (list, datasets.Dataset)):
                input_size = len(input_data)
            elif isinstance(input_data, datasets.IterableDataset):
                input_size = input_data.num_rows
        self._input_data_size = input_size

    def _on_init(self):
        stream = ({IDX_KEY: i, BODY_KEY: text} for i, text in enumerate(self._input_data))
        stream = rich.progress.track(
            stream, description=f"Building ES index {self._index_name}", total=self._input_data_size
        )
        ingest_data(stream, url=self.url, index_name=self._index_name, exist_ok=self._exist_ok)

    def _on_exit(self):
        if not self._persistent:
            client = es.Elasticsearch(self.url)
            client.indices.delete(index=self._index_name)

    def _make_cmd(self) -> list[str]:
        return ["elasticsearch"]

    @property
    def url(self):
        return f"{self.host}:{self.port}"

    def get_client(self):
        return Bm25Client(self.url, index_name=self._index_name)


def ingest_data(
    stream: Iterable[dict[str, Any]],
    *,
    url: str,
    index_name: str,
    chunk_size: int = 1000,
    exist_ok: bool = False,
):
    asyncio.run(_async_ingest_data(stream, url=url, index_name=index_name, chunk_size=chunk_size, exist_ok=exist_ok))


async def _async_ingest_data(
    stream: Iterable[dict[str, Any]],
    *,
    url: str,
    index_name: str,
    chunk_size: int = 1000,
    exist_ok: bool = False,
):
    async with es.AsyncElasticsearch(url) as client:
        if await client.indices.exists(index=index_name):
            if exist_ok:
                return
            raise RuntimeError(f"Index {index_name} already exists")
        try:
            await client.indices.create(index=index_name)
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
