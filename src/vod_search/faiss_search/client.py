import os
import sys
import time
from copy import copy
from pathlib import Path

import numpy as np
import requests
import rich
import vod_types as vt
from vod_search import base, io
from vod_search.socket import find_available_port

# get the path to the server script
server_run_path = Path(__file__).parent / "server.py"


class FaissClient(base.SearchClient):
    """Faiss client for interacting for spawning a Faiss server and querying it."""

    def __init__(
        self,
        host: str = "http://localhost",
        port: int = 7678,
    ):
        self.host = host
        self.port = port

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.url}](requires_vectors={self.requires_vectors})"

    @property
    def url(self) -> str:
        """Return the URL of the server."""
        return f"{self.host}:{self.port}"

    def ping(self, timeout: float = 120) -> bool:
        """Ping the server."""
        try:
            response = requests.get(f"{self.url}/", timeout=timeout)
        except requests.exceptions.ConnectionError:
            return False

        response.raise_for_status()
        return "OK" in response.text

    def search_py(self, query_vec: np.ndarray, top_k: int = 3, timeout: float = 120) -> vt.RetrievalBatch:
        """Search the server given a batch of vectors (slow implementation)."""
        response = requests.post(
            f"{self.url}/search",
            json={
                "vectors": query_vec.tolist(),
                "top_k": top_k,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return vt.RetrievalBatch.cast(
            indices=data["indices"],
            scores=data["scores"],
        )

    def search(
        self,
        *,
        vector: np.ndarray,
        text: None | list[str] = None,  # noqa: ARG
        ids: None | list[list[str]] = None,  # noqa: ARG
        subset_ids: None | list[list[str]] = None,  # noqa: ARG
        shard: None | list[str] = None,  # noqa: ARG
        top_k: int = 3,
        timeout: float = 120,
    ) -> vt.RetrievalBatch:
        """Search the server given a batch of vectors."""
        start_time = time.time()
        serialized_vectors = io.serialize_np_array(vector)
        payload = {
            "vectors": serialized_vectors,
            "top_k": top_k,
        }
        response = requests.post(f"{self.url}/fast-search", json=payload, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            try:
                rich.print(response.json()["detail"])
            except Exception:
                rich.print(response.text)
            raise exc

        data = response.json()
        indices_list = io.deserialize_np_array(data["indices"])
        scores_list = io.deserialize_np_array(data["scores"])

        try:
            return vt.RetrievalBatch.cast(
                indices=indices_list,
                scores=scores_list,
                labels=None,
                meta={"time": time.time() - start_time},
            )
        except Exception as exc:
            rich.print({"indices": indices_list, "scores": scores_list})
            raise exc


class FaissMaster(base.SearchMaster[FaissClient]):
    """The Faiss master client is responsible for spawning and killing the Faiss server.

    ```python
    with FaissMaster(index_path, nprobe=8, logging_level="critical") as client:
        # do stuff with the client
        result = client.search(...)
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        index_path: str | Path,
        nprobe: int = 8,
        logging_level: str = "DEBUG",
        host: str = "http://localhost",
        port: int = 6637,
        skip_setup: bool = False,
        free_resources: bool = False,
        serve_on_gpu: bool = False,
    ):
        super().__init__(skip_setup=skip_setup, free_resources=free_resources)
        self.index_path = Path(index_path)
        self.nprobe = nprobe
        self.logging_level = logging_level
        self.host = host
        if port < 0:
            port = find_available_port()
        self.port = port
        self.serve_on_gpu = serve_on_gpu

    def _make_env(self) -> dict[str, str]:
        env = copy(dict(os.environ))
        if "KMP_DUPLICATE_LIB_OK" not in env:
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["LOGURU_LEVEL"] = self.logging_level.upper()
        # add the local path, so importing the library will work.
        if "PYTHONPATH" not in env:
            env["PYTHONPATH"] = str(Path.cwd())
        else:
            os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{Path.cwd()}"
        return env

    def _make_cmd(self) -> list[str]:
        executable_path = sys.executable
        return [
            str(executable_path),
            str(server_run_path),
            "--index-path",
            str(self.index_path.absolute()),
            "--nprobe",
            str(self.nprobe),
            "--host",
            str(self.host),
            "--port",
            str(self.port),
            "--logging-level",
            str(self.logging_level),
            *(["--serve-on-gpu"] if self.serve_on_gpu else []),
        ]

    def get_client(self) -> FaissClient:
        """Get the client for interacting with the Faiss server."""
        return FaissClient(host=self.host, port=self.port)

    @property
    def url(self) -> str:
        """Return the URL of the server."""
        return f"{self.host}:{self.port}"

    @property
    def service_info(self) -> str:
        """Return the name of the service."""
        return f"FaissServer[{self.url}]"

    @property
    def service_name(self) -> str:
        """Return the name of the service."""
        return super().service_name + f"-{self.port}"
