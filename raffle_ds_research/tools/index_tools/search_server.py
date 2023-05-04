from __future__ import annotations

import abc
import copy
import os
import pathlib
import subprocess
import time
from typing import Any, Generic, Optional, TypeVar

import loguru
from typing_extensions import Self

from raffle_ds_research.tools.index_tools import retrieval_data_type as rtypes


class DoNotPickleError(Exception):
    """An exception to raise when an object cannot be pickled."""

    def __init__(self, msg: Optional[str] = None):
        msg = msg or "This object cannot be pickled."
        super().__init__(msg)


class SearchClient(abc.ABC):
    """A client to interact with a search server."""

    requires_vectors: bool = True

    @abc.abstractmethod
    def ping(self) -> bool:
        """Ping the server."""
        raise NotImplementedError()

    @abc.abstractmethod
    def search(
        self,
        *,
        text: list[str],
        vector: Optional[rtypes.Ts] = None,
        label: Optional[list[str | int]] = None,
        top_k: int = 3,
    ) -> rtypes.RetrievalBatch[rtypes.Ts]:
        """Search the server given a batch of text and/or vectors."""
        raise NotImplementedError()


Sc = TypeVar("Sc", bound=SearchClient, covariant=True)


class SearchMaster(Generic[Sc], abc.ABC):
    """A class that manages a search server."""

    _timeout: float = 180
    _server_proc: Optional[subprocess.Popen] = None
    _allow_existing_server: bool = False
    skip_setup: bool

    def __init__(self, skip_setup: bool = False) -> None:
        self.skip_setup = skip_setup

    def __enter__(self) -> "Self":
        """Start the server."""
        if not self.skip_setup:
            self._setup()
        return self

    def _setup(self) -> None:
        self._server_proc = self._start_server()
        self._on_init()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ANN401
        """Kill the server."""
        self._on_exit()
        if self._server_proc is not None:
            self._server_proc.terminate()

    def _on_init(self) -> None:
        pass

    def _on_exit(self) -> None:
        pass

    @abc.abstractmethod
    def get_client(self) -> Sc:
        """Return a client to the server."""
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
                loguru.logger.info(f"Using existing search instance {self.service_info}")
                return None
            raise RuntimeError(f"Server {self.service_name} is already running.")

        cmd = self._make_cmd()
        env = self._make_env()

        stdout_file = pathlib.Path(f"{self.service_name}.stdout.log")
        stderr_file = pathlib.Path(f"{self.service_name}.stderr.log")
        server_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_file.open("w"),
            stderr=stderr_file.open("w"),
        )

        t0 = time.time()
        loguru.logger.info(f"Spawning {self.service_info} ...")
        while not _client.ping():
            time.sleep(0.1)
            if time.time() - t0 > self._timeout:
                server_proc.terminate()
                raise TimeoutError(f"Couldn't ping the server after {self._timeout:.0f}s.")

        return server_proc

    @property
    def service_name(self) -> str:
        """Return the name of the service."""
        return self.__class__.__name__.lower()

    @property
    def service_info(self) -> str:
        """Return the name of the service."""
        return self.service_name

    def __getstate__(self) -> dict[str, Any]:
        """Prevent pickling."""
        raise DoNotPickleError(
            f"{type(self).__name__} is not pickleable. "
            f"To use in multiprocessing, using a client instead (`server.get_client()`)."
        )  # type: ignore

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Prevent unpickling."""
        raise DoNotPickleError(
            f"{type(self).__name__} is not pickleable. "
            f"To use in multiprocessing, using a client instead (`server.get_client()`)."
        )
