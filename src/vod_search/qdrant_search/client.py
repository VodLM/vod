from __future__ import annotations

import abc
import copy
import dataclasses
import itertools
import time
import uuid
import warnings
from typing import Any, Iterable, Optional

import numba
import numpy as np
import qdrant_client
import rich
from grpc import StatusCode
from grpc._channel import _InactiveRpcError
from loguru import logger
from qdrant_client.http import exceptions as qdrexc
from qdrant_client.http import models as qdrm
from rich import status
from rich.markup import escape
from rich.progress import track
from vod_search import base, rdtypes
from vod_tools import dstruct
from vod_tools.misc.pretty import human_format_nb

QDRANT_GROUP_KEY: str = "_group_"


def _init_client(host: str, port: int, grpc_port: None | int, **kwargs: Any) -> qdrant_client.QdrantClient:
    """Initialize the client."""
    try:
        return qdrant_client.QdrantClient(
            url=host,
            port=port,
            grpc_port=grpc_port or -1,
            prefer_grpc=grpc_port is not None,
        )
    except Exception as exc:
        raise Exception(
            f"Qdrant client failed to initialize. "
            f"Have you started the server at {f'{host}:{port}'}? "
            f"Start it with `docker run -p {port}:{port} qdrant/qdrant`"
        ) from exc


def _collection_name(collection_name: str, escape_rich: bool = True) -> str:
    """Format the collection name."""
    escape_fn = escape if escape_rich else lambda x: x
    return f"[bold cyan]Qdrant{escape_fn(f'[{collection_name}]')}[/bold cyan]"  # type: ignore


class QdrantSearchClient(base.SearchClient):
    """A client to interact with a search server."""

    requires_vectors: bool = True
    _index_name: str
    _client: qdrant_client.QdrantClient

    def __init__(
        self,
        host: str,
        port: int,
        grpc_port: None | int,
        index_name: str,
        search_params: qdrm.SearchParams,
        supports_groups: bool = True,
    ):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self._client = _init_client(self.host, self.port, self.grpc_port)
        self._index_name = index_name
        self.supports_groups = supports_groups
        self.search_params = search_params

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}[{self.host}:{self.port}]("
            f"requires_vectors={self.requires_vectors}, "
            f"index_name={self._index_name})"
        )

    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self._client.count(collection_name=self._index_name).count

    def ping(self) -> bool:
        """Ping the server."""
        try:
            self._client.get_collections()
            return True
        except qdrexc.UnexpectedResponse:
            return False
        except _InactiveRpcError:
            return False

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the client state."""
        state = self.__dict__.copy()
        state.pop("_client", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Recreate the client from the state."""
        self.__dict__.update(state)
        self._client = _init_client(self.host, self.port, self.grpc_port)

    def search(
        self,
        *,
        text: Optional[list[str]] = None,  # noqa: ARG002
        vector: Optional[rdtypes.Ts],
        group: Optional[list[str | int]] = None,
        section_ids: Optional[list[list[str | int]]] = None,  # noqa: ARG002
        shard: Optional[list[str]] = None,  # noqa: ARG002
        top_k: int = 3,
    ) -> rdtypes.RetrievalBatch[rdtypes.Ts]:
        """Search the server given a batch of text and/or vectors."""
        if vector is None:
            raise ValueError("vector cannot be None")
        if self.supports_groups and group is None:
            warnings.warn(f"This `{type(self).__name__}` supports group, but no label is provided.", stacklevel=2)

        def _get_filter(group: None | str | int) -> None | qdrm.Filter:
            if group is None:
                return None
            if not isinstance(group, str):
                group = int(group)

            return qdrm.Filter(
                must=[
                    qdrm.FieldCondition(
                        key=QDRANT_GROUP_KEY,
                        match=qdrm.MatchValue(value=group),
                    ),
                ],
            )

        search_result: list[list[qdrm.ScoredPoint]] = self._client.search_batch(
            collection_name=self._index_name,
            requests=[
                qdrm.SearchRequest(
                    limit=top_k,
                    vector=vector[i].tolist(),
                    filter=_get_filter(group[i]) if group is not None else None,
                    with_payload=False,
                    params=self.search_params,
                )
                for i in range(len(vector))
            ],
        )
        return _search_batch_to_rdtypes(search_result, top_k)


@numba.jit(forceobj=True, looplift=True)
def _search_batch_to_rdtypes(batch: list[list[qdrm.ScoredPoint]], top_k: int) -> rdtypes.RetrievalBatch:
    """Convert a batch of search results to rdtypes."""
    scores = np.full((len(batch), top_k), -np.inf, dtype=np.float32)
    indices = np.full((len(batch), top_k), -1, dtype=np.int64)
    max_j = -1
    for i, row in enumerate(batch):
        for j, p in enumerate(row):
            scores[i, j] = p.score
            indices[i, j] = p.id
            if j > max_j:
                max_j = j

    return rdtypes.RetrievalBatch(
        scores=scores[:, : max_j + 1],
        indices=indices[:, : max_j + 1],
    )


@dataclasses.dataclass(frozen=True)
class IndexValidationStatus:
    """Validation status of an index."""

    valid: bool
    message: str


class QdrantSearchMaster(base.SearchMaster[QdrantSearchClient], abc.ABC):
    """A class that manages a search server."""

    _allow_existing_server: bool = True

    def __init__(
        self,
        vectors: dstruct.SizedDataset[np.ndarray],
        *,
        groups: Optional[Iterable[str | int]] = None,
        host: str = "http://localhost",
        port: int = 6333,
        grpc_port: None | int = 6334,
        index_name: Optional[str] = None,
        persistent: bool = True,
        exist_ok: bool = True,
        skip_setup: bool = False,
        qdrant_body: Optional[dict[str, Any]] = None,
        search_params: Optional[dict[str, Any]] = None,
        free_resources: bool = False,
        force_single_collection: bool = False,
    ) -> None:
        super().__init__(skip_setup=skip_setup, free_resources=free_resources)
        self._host = host
        self._port = port
        self._grpc_port = grpc_port
        self._vectors = vectors
        self._input_groups = groups
        if index_name is None:
            index_name = f"auto-{uuid.uuid4().hex}"
        self._index_name = index_name
        self._persistent = persistent
        self._exist_ok = exist_ok
        self._qdrant_body = qdrant_body
        self._search_params = qdrm.SearchParams(**(search_params or {}))
        self._force_single_collection = force_single_collection

    @property
    def supports_groups(self) -> bool:
        """Whether the index supports labels."""
        return self._input_groups is not None

    def get_client(self) -> QdrantSearchClient:
        """Return a client to the server."""
        return QdrantSearchClient(
            host=self._host,
            port=self._port,
            grpc_port=self._grpc_port,
            index_name=self._index_name,
            supports_groups=self.supports_groups,
            search_params=self._search_params,
        )

    def _make_cmd(self) -> list[str]:
        return [
            "docker",
            "run",
            "-p",
            f"{self._port}:{self._port}",
            "-p",
            f"{self._grpc_port}:{self._grpc_port}",
            "qdrant/qdrant",
        ]

    def _make_env(self) -> dict[str, Any]:
        env = super()._make_env()
        env["QDRANT__SERVICE__HTTP_PORT"] = str(self._port)
        if self._grpc_port is not None:
            env["QDRANT__SERVICE__GRPC_PORT"] = str(self._grpc_port)
        return env

    def _on_init(self) -> None:
        client = _init_client(self._host, self._port, self._grpc_port)
        vshp = self._vectors[0].shape
        if len(vshp) != 1:
            raise ValueError(f"Expected a 1D vectors, got {vshp}")

        # Delete other collections
        if self._force_single_collection:
            _delete_except([self._index_name], client)

        # Check wheter the index exist
        index_exist = _collection_exists(client, self._index_name)
        if index_exist and not self._exist_ok:
            raise FileNotFoundError(f"{_collection_name(self._index_name, escape_rich=False)}: already exists.")

        # Validate the index and delete if necessary
        if index_exist:
            valid_status = _validate(client, self._index_name, self._vectors, raise_if_invalid=True)
            if not valid_status.valid:
                logger.warning(
                    f"{_collection_name(self._index_name, escape_rich=False)}: "
                    f"already exists but is not valid ({valid_status.message}). "
                )
                client.delete_collection(collection_name=self._index_name)
                index_exist = False

        if not index_exist:
            # Create the index & Ingest the data
            body = _make_qdrant_body(vshp[-1], self._qdrant_body)
            rich.print(f">> [magenta bold]Creating collection: {self._index_name}")
            client.recreate_collection(collection_name=self._index_name, **body)

            with DisableIndexing(client, self._index_name, delete_on_exception=True):
                _ingest_data(
                    client=client,
                    collection_name=self._index_name,
                    vectors=self._vectors,
                    groups=self._input_groups,
                )

            # Validate the index
            _validate(client, self._index_name, self._vectors, raise_if_invalid=True)

    def _on_exit(self) -> None:
        if not self._persistent:
            client = _init_client(self._host, self._port, self._grpc_port)
            client.delete_collection(collection_name=self._index_name)

    def _free_resources(self) -> None:
        client = _init_client(self._host, self._port, self._grpc_port)
        with status.Status(f"{_collection_name(self._index_name)}: Deleting other indices.."):
            _delete_except([self._index_name], client)
            while not self.get_client().ping():
                time.sleep(0.05)


def _validate(
    client: qdrant_client.QdrantClient,
    collection_name: str,
    vectors: dstruct.SizedDataset[np.ndarray],
    raise_if_invalid: bool = False,
) -> IndexValidationStatus:
    """Validate the index."""
    with status.Status(f"{_collection_name(collection_name)}: Validating.."):
        while client.get_collection(collection_name=collection_name).status != qdrm.CollectionStatus.GREEN:
            time.sleep(0.05)
        count = client.count(collection_name=collection_name).count
        if count != len(vectors):
            valid_status = IndexValidationStatus(
                valid=False,
                message=(
                    f"{_collection_name(collection_name, escape_rich=False)}: "
                    f"Number of vectors doesn't match: {count} != {len(vectors)}"
                ),
            )
            if raise_if_invalid:
                raise Exception(valid_status.message)
            return valid_status
    return IndexValidationStatus(valid=True, message="")


def _delete_except(exclude_list: list[str], client: qdrant_client.QdrantClient) -> None:
    for col in client.get_collections().collections:
        if col.name not in exclude_list:
            logger.debug(f"Deleting collection `{col.name}`")
            client.delete_collection(collection_name=col.name)


def _collection_exists(client: qdrant_client.QdrantClient, collection_name: str) -> bool:
    try:
        client.get_collection(collection_name=collection_name)
        index_exist = True
    except qdrexc.UnexpectedResponse as exc:
        if exc.status_code != 404:  # noqa: PLR2004
            raise Exception(f"Unexpected error: {exc}") from exc
        rich.print(exc)
        index_exist = False

    except _InactiveRpcError as exc:
        if exc.code() != StatusCode.NOT_FOUND:
            raise Exception(f"Unexpected error: {exc}") from exc
        rich.print(exc)
        index_exist = False
    return index_exist


class DisableIndexing:
    """Temporarily disable indexing."""

    _opt_config: None | qdrm.OptimizersConfig

    def __init__(
        self, client: qdrant_client.QdrantClient, collection_name: str, delete_on_exception: bool = True
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._opt_config = None
        self.delete_on_exception = delete_on_exception

    def __enter__(self) -> None:
        collection_info = self._client.get_collection(collection_name=self._collection_name)
        self._opt_config = collection_info.config.optimizer_config
        if self._opt_config is None:
            raise Exception(f"No optimizer config for collection `{self._collection_name}`")
        self._client.update_collection(
            collection_name=self._collection_name,
            optimizers_config={"indexing_threshold": 0},  # type: ignore
        )

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:  # noqa: ANN401
        if exc_type is not None and self.delete_on_exception:
            self._client.delete_collection(collection_name=self._collection_name)
            raise Exception(f"Collection `{self._collection_name}` deleted due to {exc_type}") from exc_value

        with status.Status(f"{_collection_name(self._collection_name)}: indexing..."):
            self._client.update_collection(
                collection_name=self._collection_name,
                optimizers_config=self._opt_config.dict(),  # type: ignore
            )


def _make_qdrant_body(dim: int, body: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Make the default Qdrant configuration body."""
    body = copy.deepcopy(body) or {}
    body["vectors_config"] = qdrm.VectorParams(
        **{
            "size": int(dim),
            "distance": qdrm.Distance.DOT,
            **body.get("vectors_config", {}),
        }
    )
    if "quantization_config" in body:
        quant_conf = body.pop("quantization_config")
        keys = set(quant_conf.keys())
        if len(keys) > 1:
            raise ValueError(f"Only one quantization method is supported, got {keys}")
        key = keys.pop()
        args = quant_conf[key]
        sc_cfg, cfg = {
            "scalar": (qdrm.ScalarQuantization, qdrm.ScalarQuantizationConfig),
            "product": (qdrm.ProductQuantization, qdrm.ProductQuantizationConfig),
        }[key]
        body["quantization_config"] = sc_cfg(**{key: cfg(**args)})

    return body


def _ingest_data(
    client: qdrant_client.QdrantClient,
    collection_name: str,
    vectors: dstruct.SizedDataset[np.ndarray],
    groups: Optional[Iterable[str | int]] = None,
    batch_size: int = 1_000,
) -> None:
    """Ingest the data (vectors & groups) into the index."""
    groups_iter = iter(groups) if groups is not None else itertools.repeat(None)
    for j in track(
        range(0, len(vectors), batch_size),
        description=f"{_collection_name(collection_name)}: Ingesting {human_format_nb(len(vectors))} vectors",
    ):
        vec_chunk = vectors[j : j + batch_size]
        if groups is None:
            payloads = None
        else:
            group_chunk = [next(groups_iter) for _ in range(len(vec_chunk))]
            payloads = [{QDRANT_GROUP_KEY: g if isinstance(g, str) else int(g)} for g in group_chunk]  # type: ignore
        ids = np.arange(j, j + len(vec_chunk))
        batch = qdrm.Batch(
            ids=ids.tolist(),
            vectors=vec_chunk.tolist(),
            payloads=payloads,
        )
        client.upsert(
            collection_name=collection_name,
            wait=False,  # TODO: how to explicitely wait for the indexing to finish?
            points=batch,
        )
