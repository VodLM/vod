from __future__ import annotations

import os
import subprocess
import sys
import time
from copy import copy
from pathlib import Path
from typing import Optional, Type

import numpy as np
import requests
import torch

from raffle_ds_research.tools.index_tools import io
from raffle_ds_research.tools.index_tools.retrieval_data_type import RetrievalBatch, RetrievalDataType, Ts

# get the path to the server script
server_run_path = Path(__file__).parent / "server.py"


class FaissClient(object):
    """Faiss client for interacting for spawning a Faiss server and querying it."""

    def __init__(
        self,
        host: str = "http://localhost",
        port: int = 7678,
    ):
        self.host = host
        self.port = port

    @property
    def url(self):
        return f"{self.host}:{self.port}"

    def ping(self) -> str:
        response = requests.get(f"{self.url}/")
        response.raise_for_status()
        return response.text

    def search_py(self, query_vec: Ts, top_k: int = 3) -> RetrievalBatch[Ts]:
        input_type = type(query_vec)
        response = requests.post(
            f"{self.url}/search",
            json={
                "vectors": query_vec.tolist(),
                "top_k": top_k,
            },
        )
        response.raise_for_status()
        data = response.json()
        indices_list = data["indices"]
        scores_list = data["scores"]
        cast_fn = {
            torch.Tensor: torch.tensor,
            np.ndarray: np.array,
        }[input_type]
        indices = cast_fn(indices_list)
        scores = cast_fn(scores_list)
        return RetrievalBatch(indices=indices, scores=scores)

    def search(self, query_vec: Ts, top_k: int = 3) -> RetrievalBatch[Ts]:
        input_type = type(query_vec)
        input_type_enum, serialized_fn = {
            torch.Tensor: (RetrievalDataType.TORCH, io.serialize_torch_tensor),
            np.ndarray: (RetrievalDataType.NUMPY, io.serialize_np_array),
        }[input_type]
        serialized_vectors = serialized_fn(query_vec)
        response = requests.post(
            f"{self.url}/fast-search",
            json={
                "vectors": serialized_vectors,
                "top_k": top_k,
                "array_type": input_type_enum.value,
            },
        )
        response.raise_for_status()
        data = response.json()
        encoded_indices = data["indices"]
        encoded_scores = data["scores"]
        return decode_faiss_results(indices=encoded_indices, scores=encoded_scores, target_type=input_type)


class FaissMaster(object):
    """The Faiss master client is responsible for spawning and killing the Faiss server.

    ```python
    with FaissMaster(index_path, nprobe=8, logging_level="critical") as faiss_master:
        client = faiss_master.get_client()

        # do stuff with the client
        result = client.search(...)
    ```
    """

    _server_proc: Optional[subprocess.Popen] = None

    def __init__(
        self,
        index_path: str | Path,
        nprobe: int = 8,
        logging_level: str = "CRITICAL",
        log_dir: Path = None,
        host: str = "http://localhost",
        port: int = 7678,
    ):
        self.index_path = Path(index_path)
        self.nprobe = nprobe
        self.logging_level = logging_level
        self.log_dir = log_dir
        self.host = host
        self.port = port

    def __enter__(self) -> "FaissMaster":
        self._server_proc = self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._server_proc.kill()

    def _start_server(self) -> subprocess.Popen:
        if self._server_proc is not None:
            raise RuntimeError("Server already running.")
        cmd = self._make_cmd()
        env = self._make_env()
        server_proc = subprocess.Popen(cmd, env=env)
        client = self.get_client()
        while True:
            try:
                health_check = client.ping()
                if "ok" not in health_check.lower():
                    msg = f"Server health check failed: {health_check}"
                    raise RuntimeError(msg)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)
                continue

        return server_proc

    def _make_env(self) -> dict[str, str]:
        env = copy(dict(os.environ))
        if "KMP_DUPLICATE_LIB_OK" not in env:
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["LOGURU_LEVEL"] = self.logging_level.upper()
        return env

    def _make_cmd(self) -> list[str]:
        executable_path = sys.executable
        cmd = [
            executable_path,
            server_run_path,
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
        ]
        if self.log_dir is not None:
            cmd.extend(["--log-dir", str(self.log_dir.absolute())])
        return cmd

    def get_client(self) -> FaissClient:
        return FaissClient(host=self.host, port=self.port)

    def __getstate__(self):
        raise DoNotPickleError(
            "FaissMaster is not pickleable. To use in multiprocessing, using a client instead (`master.get_client()`)."
        )

    def __setstate__(self, state):
        raise DoNotPickleError(
            "FaissMaster is not pickleable. To use in multiprocessing, using a client instead (`master.get_client()`)."
        )


class DoNotPickleError(Exception):
    def __init__(self, msg: Optional[str] = None):
        msg = msg or "This object is not pickleable."
        super().__init__(msg)


def decode_faiss_results(*, indices: str, scores: str, target_type: Type[Ts]) -> RetrievalBatch[Ts]:
    indices = io.deserialize_np_array(indices)
    scores = io.deserialize_np_array(scores)
    if target_type == torch.Tensor:
        indices = torch.from_numpy(indices)
        scores = torch.from_numpy(scores)
    return RetrievalBatch(indices=indices, scores=scores)
