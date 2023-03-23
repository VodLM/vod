from __future__ import annotations

import os
import sys
from copy import copy
from pathlib import Path
from typing import Optional, Type

import aiohttp
import numpy as np
import requests
import rich
import torch

from raffle_ds_research.tools.index_tools import io
from raffle_ds_research.tools.index_tools import retrieval_data_type as rtypes
from raffle_ds_research.tools.index_tools import search_server

# get the path to the server script
server_run_path = Path(__file__).parent / "server.py"


class FaissClient(search_server.SearchClient):
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

    def ping(self) -> bool:
        try:
            response = requests.get(f"{self.url}/")
        except requests.exceptions.ConnectionError:
            return False

        response.raise_for_status()
        return "OK" in response.text

    def search_py(self, query_vec: rtypes.Ts, top_k: int = 3) -> rtypes.RetrievalBatch[rtypes.Ts]:
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
        return rtypes.RetrievalBatch(indices=indices, scores=scores)

    def search(
        self, *, vector: rtypes.Ts, text: Optional[list[str]] = None, top_k: int = 3
    ) -> rtypes.RetrievalBatch[rtypes.Ts]:
        input_type = type(vector)
        input_type_enum, serialized_fn = {
            torch.Tensor: (rtypes.RetrievalDataType.TORCH, io.serialize_torch_tensor),
            np.ndarray: (rtypes.RetrievalDataType.NUMPY, io.serialize_np_array),
        }[input_type]
        serialized_vectors = serialized_fn(vector)
        payload = {
            "vectors": serialized_vectors,
            "top_k": top_k,
            "array_type": input_type_enum.value,
        }
        response = requests.post(f"{self.url}/fast-search", json=payload)
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
        cast_fn = {
            torch.Tensor: torch.tensor,
            np.ndarray: np.array,
        }[input_type]
        indices = cast_fn(indices_list)
        scores = cast_fn(scores_list)
        try:
            return rtypes.RetrievalBatch(indices=indices, scores=scores)
        except Exception as exc:
            rich.print({"indices": indices, "scores": scores})
            raise exc


class FaissMaster(search_server.SearchMaster[FaissClient]):
    """The Faiss master client is responsible for spawning and killing the Faiss server.

    ```python
    with FaissMaster(index_path, nprobe=8, logging_level="critical") as client:
        # do stuff with the client
        result = client.search(...)
    ```
    """

    def __init__(
        self,
        index_path: str | Path,
        nprobe: int = 8,
        logging_level: str = "CRITICAL",
        host: str = "http://localhost",
        port: int = 7678,
    ):
        self.index_path = Path(index_path)
        self.nprobe = nprobe
        self.logging_level = logging_level
        self.host = host
        self.port = port

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
        return cmd

    def get_client(self) -> FaissClient:
        return FaissClient(host=self.host, port=self.port)


def decode_faiss_results(
    *, indices: str, scores: str, target_type: Type[rtypes.Ts]
) -> rtypes.RetrievalBatch[rtypes.Ts]:
    indices = io.deserialize_np_array(indices)
    scores = io.deserialize_np_array(scores)
    if target_type == torch.Tensor:
        indices = torch.from_numpy(indices)
        scores = torch.from_numpy(scores)
    return rtypes.RetrievalBatch(indices=indices, scores=scores)
