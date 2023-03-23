import abc
import copy
import os
import subprocess
import time
from typing import Any, Generic, Optional, TypeVar

import loguru
import rich.status

from raffle_ds_research.tools.index_tools import retrieval_data_type as rtypes


class DoNotPickleError(Exception):
    def __init__(self, msg: Optional[str] = None):
        msg = msg or "This object cannot be pickled."
        super().__init__(msg)


class SearchClient(abc.ABC):
    requires_vectors: bool = True

    @abc.abstractmethod
    def ping(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def search(
        self,
        *,
        text: list[str],
        vector: rtypes.Ts,
        top_k: int = 3,
    ) -> rtypes.RetrievalBatch[rtypes.Ts]:
        raise NotImplementedError()


Sc = TypeVar("Sc", bound=SearchClient)


class SearchMaster(Generic[Sc], abc.ABC):
    _timeout: float = 180
    _server_proc: Optional[subprocess.Popen] = None
    _allow_existing_server: bool = False

    def __enter__(self) -> "Self":
        self._server_proc = self._start_server()
        self._on_init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._on_exit()
        if self._server_proc is not None:
            self._server_proc.terminate()

    def _on_init(self):
        pass

    def _on_exit(self):
        pass

    @abc.abstractmethod
    def get_client(self) -> Sc:
        raise NotImplementedError

    @abc.abstractmethod
    def _make_cmd(self) -> list[str]:
        raise NotImplementedError

    def _make_env(self) -> dict[str, Any]:
        return copy.copy(dict(os.environ))

    def _start_server(self) -> Optional[subprocess.Popen]:
        _client = self.get_client()
        if _client.ping():
            if self._allow_existing_server:
                loguru.logger.info(f"Using existing server {self.service_name}.")
                return None
            else:
                raise RuntimeError(f"Server {self.service_name} is already running.")

        cmd = self._make_cmd()
        env = self._make_env()

        server_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=open(f"{self.service_name}.stdout.log", "w"),
            stderr=open(f"{self.service_name}.es.stderr.log", "w"),
        )

        t0 = time.time()
        with rich.status.Status(f"Waiting for server {self.service_name} to start..."):
            while not _client.ping():
                time.sleep(0.1)
                if time.time() - t0 > self._timeout:
                    server_proc.terminate()
                    raise TimeoutError(f"Couldn't ping the server after {self._timeout:.0f}s.")

        return server_proc

    @property
    def service_name(self) -> str:
        return self.__class__.__name__.lower()

    def __getstate__(self):
        raise DoNotPickleError(
            f"{type(self.__name__)} is not pickleable. "
            f"To use in multiprocessing, using a client instead (`server.get_client()`)."
        )

    def __setstate__(self, state):
        raise DoNotPickleError(
            f"{type(self.__name__)} is not pickleable. "
            f"To use in multiprocessing, using a client instead (`server.get_client()`)."
        )
