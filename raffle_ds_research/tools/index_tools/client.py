import os
import subprocess
import sys
import time
from copy import copy
from pathlib import Path
from typing import Type

import numpy as np
import requests
import torch

from raffle_ds_research.tools.index_tools import io
from raffle_ds_research.tools.index_tools.data_models import FaissResults, Ts, IoArrayType
from raffle_ds_research.tools.index_tools.io import deserialize_np_array

# get the path to the server script
server_run_path = Path(__file__).parent / "server.py"


class FaissClient(object):
    """Faiss client for interacting for spawning a Faiss server and querying it."""

    def __init__(
        self,
        index_path: Path,
        nprobe: int = 8,
        host: str = "http://localhost",
        port: int = 7678,
        logging_level: str = "CRITICAL",
        log_dir: Path = None,
    ):
        self.index_path = index_path
        self.host = host
        self.port = port
        self.nprobe = nprobe
        self.server_proc = None
        self.logging_level = logging_level
        self.log_dir = log_dir  # todo: handle logs

    @property
    def url(self):
        return f"{self.host}:{self.port}"

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.server_proc.kill()

    def start_server(self):
        cmd = self._make_cmd()
        self.server_proc = subprocess.Popen(cmd, env=copy(os.environ))
        while True:
            try:
                health_check = self.ping()
                if "ok" not in health_check.lower():
                    msg = f"Server health check failed: {health_check}"
                    raise RuntimeError(msg)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(0.05)
                continue

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

    def ping(self) -> str:
        response = requests.get(f"{self.url}/")
        response.raise_for_status()
        return response.text

    def search_py(self, query_vec: Ts, top_k: int = 3) -> FaissResults[Ts]:
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
        return FaissResults(indices=indices, scores=scores)

    def search(self, query_vec: Ts, top_k: int = 3) -> FaissResults[Ts]:
        input_type = type(query_vec)
        input_type_enum, serialized_fn = {
            torch.Tensor: (IoArrayType.TORCH, io.serialize_torch_tensor),
            np.ndarray: (IoArrayType.NUMPY, io.serialize_np_array),
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


def decode_faiss_results(*, indices: str, scores: str, target_type: Type[Ts]) -> FaissResults[Ts]:
    indices = deserialize_np_array(indices)
    scores = deserialize_np_array(scores)
    if target_type == torch.Tensor:
        indices = torch.from_numpy(indices)
        scores = torch.from_numpy(scores)
    return FaissResults(indices=indices, scores=scores)
