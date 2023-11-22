import abc
import copy
import os
import pathlib
import subprocess
import time
import typing as typ

import loguru
import numpy as np
import vod_types as vt
from typing_extensions import Self

ShardName: typ.TypeAlias = str
SubsetId: typ.TypeAlias = str
SectionId: typ.TypeAlias = str


def _camel_to_snake(name: str) -> str:
    """Convert a camel case name to snake case."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


class DoNotPickleError(Exception):
    """An exception to raise when an object cannot be pickled."""

    def __init__(self, msg: None | str = None):
        msg = msg or "This object cannot be pickled."
        super().__init__(msg)


class SearchClient(abc.ABC):
    """A client to interact with a search server."""

    requires_vectors: bool = True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(requires_vectors={self.requires_vectors})"

    @abc.abstractmethod
    def ping(self) -> bool:
        """Ping the server."""
        raise NotImplementedError()

    @abc.abstractmethod
    def search(
        self,
        *,
        text: list[str],
        vector: None | np.ndarray = None,
        subset_ids: None | list[list[SubsetId]] = None,
        ids: None | list[list[SectionId]] = None,
        shard: None | list[ShardName] = None,
        top_k: int = 3,
    ) -> vt.RetrievalBatch:
        """Search the server given a batch of text and/or vectors."""
        raise NotImplementedError()

    async def async_search(
        self,
        *,
        text: list[str],
        vector: None | np.ndarray = None,
        subset_ids: None | list[list[SubsetId]] = None,
        ids: None | list[list[SectionId]] = None,
        shard: None | list[ShardName] = None,
        top_k: int = 3,
    ) -> vt.RetrievalBatch:
        """Search the server given a batch of text and/or vectors."""
        return self.search(
            text=text,
            vector=vector,
            subset_ids=subset_ids,
            ids=ids,
            shard=shard,
            top_k=top_k,
        )


Sc_co = typ.TypeVar("Sc_co", bound=SearchClient, covariant=True)


class SearchMaster(typ.Generic[Sc_co], abc.ABC):
    """A class that manages a search server."""

    _timeout: float = 300
    _server_proc: None | subprocess.Popen = None
    _allow_existing_server: bool = False
    skip_setup: bool
    free_resources: bool

    def __init__(
        self,
        skip_setup: bool = False,
        free_resources: bool = False,
    ) -> None:
        self.skip_setup = skip_setup
        self.free_resources = free_resources

    def __enter__(self) -> "Self":
        """Start the server."""
        if self.free_resources:
            self._free_resources()
        if not self.skip_setup:
            self._setup()
        return self

    def _setup(self) -> None:
        self._server_proc = self._start_server()
        self._on_init()

    def __exit__(self, exc_type: typ.Any, exc_val: typ.Any, exc_tb: typ.Any) -> None:  # noqa: ANN401
        """Kill the server."""
        self._on_exit()
        if self._server_proc is not None:
            self._server_proc.terminate()

    def _free_resources(self) -> None:
        pass

    def _on_init(self) -> None:
        pass

    def _on_exit(self) -> None:
        pass

    @abc.abstractmethod
    def get_client(self) -> Sc_co:
        """Return a client to the server."""
        raise NotImplementedError

    @abc.abstractmethod
    def _make_cmd(self) -> list[str]:
        raise NotImplementedError

    def _make_env(self) -> dict[str, typ.Any]:
        return copy.copy(dict(os.environ))

    def _start_server(self) -> None | subprocess.Popen:
        _client = self.get_client()
        if _client.ping():
            if self._allow_existing_server:
                loguru.logger.debug(f"Connecting to existing {self.service_info}")
                return None
            raise RuntimeError(f"Server {self.service_name} is already running.")

        cmd = self._make_cmd()
        env = self._make_env()

        # setup log files
        stdout_file = pathlib.Path(f"{self.service_name}.stdout.log")
        loguru.logger.debug(f"Writing stdout to `{stdout_file.absolute()}`")
        if stdout_file.exists():
            stdout_file.unlink()
        stderr_file = pathlib.Path(f"{self.service_name}.stderr.log")
        loguru.logger.debug(f"Writing stderr to `{stderr_file.absolute()}`")
        if stderr_file.exists():
            stderr_file.unlink()

        # spawn server
        server_proc = subprocess.Popen(
            cmd,  # noqa: S603
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
        loguru.logger.debug(f"Spawned {self.service_info} in {time.time() - t0:.1f}s.")
        return server_proc

    @property
    def service_name(self) -> str:
        """Return the name of the service."""
        return _camel_to_snake(self.__class__.__name__)

    @property
    def service_info(self) -> str:
        """Return the name of the service."""
        return self.service_name

    def __getstate__(self) -> dict[str, typ.Any]:
        """Prevent pickling."""
        raise DoNotPickleError(
            f"{type(self).__name__} is not pickleable. "
            f"To use in multiprocessing, using a client instead (`server.get_client()`)."
        )  # type: ignore

    def __setstate__(self, state: dict[str, typ.Any]) -> None:
        """Prevent unpickling."""
        raise DoNotPickleError(
            f"{type(self).__name__} is not pickleable. "
            f"To use in multiprocessing, using a client instead (`server.get_client()`)."
        )
