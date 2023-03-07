import os
import subprocess
import sys
import time
from copy import copy
from pathlib import Path

import numpy as np
import requests

from raffle_ds_research.tools.index_tools.data_models import FaissNumpyResults

# get the path to the server script
server_run_path = Path(__file__).parent / "server.py"


class FaissClient(object):
    """Faiss client for interacting for spawning a Faiss server and querying it."""

    def __init__(
        self,
        index_path: Path,
        nprobe: int = 8,
        base_url: str = "http://localhost",
        port: int = 7678,
        log_dir: Path = None,
    ):
        self.index_path = index_path
        self.base_url = base_url
        self.port = port
        self.nprobe = nprobe
        self.server_proc = None
        self.log_dir = log_dir  # todo: handle logs

    @property
    def url(self):
        return f"{self.base_url}:{self.port}"

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.server_proc.kill()

    def start_server(self):
        executable_path = sys.executable
        cmd = [
            executable_path,
            server_run_path,
            "--index-path",
            str(self.index_path.absolute()),
            "--nprobe",
            str(self.nprobe),
        ]
        env = copy(os.environ)
        self.server_proc = subprocess.Popen(cmd, env=env)
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

    def ping(self) -> str:
        response = requests.get(f"{self.url}/")
        response.raise_for_status()
        return response.text

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> FaissNumpyResults:
        response = requests.post(
            f"{self.url}/search",
            json={
                "vectors": query_vec.tolist(),
                "top_k": top_k,
            },
        )
        response.raise_for_status()
        return FaissNumpyResults(**response.json())
