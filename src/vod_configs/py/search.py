import copy
from typing import Literal, Optional, TypeVar, Union

import faiss
import omegaconf
import pydantic
import torch
from loguru import logger
from typing_extensions import Self, Type
from vod_configs.py.utils import StrictModel
from vod_tools.misc.config import as_pyobj_validator

from src.vod_tools import pipes
from src.vod_tools.misc.pretty import human_format_bytes

try:
    from faiss import GpuMultipleClonerOptions, GpuResources, StandardGpuResources  # type: ignore
except ImportError:
    GpuMultipleClonerOptions = None
    GpuResources = None
    StandardGpuResources = None

FAISS_METRICS = {
    "l2": faiss.METRIC_L2,
    "inner_product": faiss.METRIC_INNER_PRODUCT,
    "l1": faiss.METRIC_L1,
    "linf": faiss.METRIC_Linf,
    "js": faiss.METRIC_JensenShannon,
}

FAISS_METRICS_INV = {v: k for k, v in FAISS_METRICS.items()}


def _get_gpu_resources(
    devices: list[int], tempmem: int = -1, log_mem_allocation: bool = False
) -> list[GpuResources]:  # type: ignore
    """Return a list of GPU resources."""
    gpu_resources = []
    ngpu = torch.cuda.device_count() if devices is None else len(devices)
    for i in range(ngpu):
        res = StandardGpuResources()  # type: ignore
        res.setLogMemoryAllocations(log_mem_allocation)
        if tempmem is not None and tempmem > 0:
            logger.debug(f"Setting GPU:{i} temporary memory to {human_format_bytes(tempmem, 'MB')}")
            res.setTempMemory(tempmem)

        gpu_resources.append(res)

    return gpu_resources


class FaissGpuConfig(StrictModel):
    """Configuration for training a faiss index on GPUs."""

    devices: list[int] = [-1]
    use_float16: bool = True
    use_precomputed_tables: bool = True
    max_add: Optional[int] = 2**18
    tempmem: Optional[int] = -1
    keep_indices_on_cpu: bool = False
    verbose: bool = True
    shard: bool = True
    add_batch_size: int = 2**18

    @pydantic.validator("devices", pre=True, always=True)
    def _validate_devices(cls, v):  # noqa: ANN
        if v is None or v == [-1]:
            return list(range(torch.cuda.device_count()))
        return v

    def cloner_options(self) -> GpuMultipleClonerOptions:  # type: ignore
        """Return a faiss.GpuMultipleClonerOptions."""
        co = GpuMultipleClonerOptions()  # type: ignore
        co.useFloat16 = self.use_float16
        co.useFloat16CoarseQuantizer = False
        co.usePrecomputed = self.use_precomputed_tables
        if self.keep_indices_on_cpu:
            co.indicesOptions = faiss.INDICES_CPU  # type: ignore
        co.verbose = self.verbose
        if self.max_add is not None:
            co.reserveVecs = self.max_add

        co.shard = self.shard

        return co

    def gpu_resources(self) -> list[GpuResources]:  # type: ignore
        """Return a list of GPU resources."""
        if not self.devices:
            raise ValueError(f"devices must be set to use `resource_vectors()`. devices={self.devices}")
        return _get_gpu_resources(self.devices, self.tempmem or -1)


SearchBackends = Literal["elasticsearch", "faiss", "qdrant"]


class BaseSearchFactoryConfig(StrictModel):
    """Base config for all search engines."""

    backend: SearchBackends = pydantic.Field(..., description="Search backend to use.")
    text_key: str = pydantic.Field("text", description="Text field to be indexed.")
    group_key: Optional[str] = pydantic.Field("group", description="Group field to be indexed.")
    section_id_key: Optional[str] = pydantic.Field("id", description="Section ID field to be indexed.")


class BaseSearchFactoryDiff(StrictModel):
    """Relative search configs."""

    backend: SearchBackends
    text_key: Optional[str] = None
    group_key: Optional[str] = None
    section_id_key: Optional[str] = None


class FaissFactoryDiff(BaseSearchFactoryDiff):
    """Configures a relative faiss configuration."""

    backend: Literal["faiss"] = "faiss"
    factory: Optional[str] = None
    nprobe: Optional[int] = None
    metric: Optional[int] = None
    train_size: Optional[int] = None
    logging_level: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    gpu: Optional[FaissGpuConfig] = None


class FaissFactoryConfig(BaseSearchFactoryConfig):
    """Configures the building of a faiss server."""

    backend: Literal["faiss"] = "faiss"
    factory: str = "Flat"
    nprobe: int = 16
    metric: int = faiss.METRIC_INNER_PRODUCT
    train_size: Optional[int] = None
    logging_level: str = "DEBUG"
    host: str = "http://localhost"
    port: int = -1
    gpu: Optional[FaissGpuConfig] = None

    def __add__(self, diff: None | FaissFactoryDiff) -> Self:
        if diff is None:
            return self
        diffs = {k: v for k, v in diff if v is not None}
        return self.copy(update=diffs)

    @pydantic.validator("metric", pre=True)
    def _validate_metric(cls, v: str | int) -> int:
        if isinstance(v, int):
            return v

        return FAISS_METRICS[v]

    def fingerprint(self) -> str:
        """Return a fingerprint for this config."""
        excludes = {"host", "port", "logging_level"}
        return pipes.fingerprint(self.dict(exclude=excludes))


class ElasticsearchFactoryDiff(BaseSearchFactoryDiff):
    """Configures a relative elasticsearch configuration."""

    backend: Literal["elasticsearch"] = "elasticsearch"
    host: Optional[str] = None
    port: Optional[int] = None
    persistent: Optional[bool] = None
    es_body: Optional[dict] = None


class ElasticsearchFactoryConfig(BaseSearchFactoryConfig):
    """Configures the building of an Elasticsearch server."""

    backend: Literal["elasticsearch"] = "elasticsearch"
    host: str = "http://localhost"
    port: int = 9200
    persistent: bool = True
    es_body: Optional[dict] = None

    def __add__(self, diff: None | ElasticsearchFactoryDiff) -> Self:
        if diff is None:
            return self
        diffs = {k: v for k, v in diff if v is not None}
        return self.copy(update=diffs)

    @pydantic.validator("es_body", pre=True)
    def _validate_es_body(cls, v: dict | None) -> dict | None:
        if isinstance(v, omegaconf.DictConfig):
            v = omegaconf.OmegaConf.to_container(v, resolve=True)  # type: ignore
        return v

    def fingerprint(self) -> str:
        """Return a fingerprint for this config."""
        excludes = {"host", "port", "persistent"}
        return pipes.fingerprint(self.dict(exclude=excludes))


class QdrantFactoryDiff(BaseSearchFactoryDiff):
    """Configures a relative qdrant configuration."""

    backend: Literal["qdrant"] = "qdrant"
    host: Optional[str] = None
    port: Optional[int] = None
    grpc_port: Optional[int] = None
    persistent: Optional[bool] = None
    exist_ok: Optional[bool] = None
    qdrant_body: Optional[dict] = None
    search_params: Optional[dict] = None
    force_single_collection: Optional[bool] = None


class QdrantFactoryConfig(BaseSearchFactoryConfig):
    """Configures the building of a Qdrant server."""

    backend: Literal["qdrant"] = "qdrant"
    host: str = "http://localhost"
    port: int = 6333
    grpc_port: Optional[int] = 6334
    persistent: bool = False
    exist_ok: bool = True
    qdrant_body: Optional[dict] = None
    search_params: Optional[dict] = None
    force_single_collection: bool = False

    def __add__(self, diff: None | QdrantFactoryDiff) -> Self:
        if diff is None:
            return self
        diffs = {k: v for k, v in diff if v is not None}
        return self.copy(update=diffs)

    @pydantic.validator("qdrant_body", pre=True)
    def _validate_qdrant_body(cls, v: dict | None) -> dict | None:
        if isinstance(v, omegaconf.DictConfig):
            v = omegaconf.OmegaConf.to_container(v, resolve=True)  # type: ignore
        return v

    @pydantic.validator("search_params", pre=True)
    def _validate_search_params(cls, v: dict | None) -> dict | None:
        if isinstance(v, omegaconf.DictConfig):
            v = omegaconf.OmegaConf.to_container(v, resolve=True)  # type: ignore
        return v

    def fingerprint(self) -> str:
        """Return a fingerprint for this config."""
        excludes = {
            "host",
            "port",
            "grpc_port",
            "persistent",
            "force_single_collection",
            "search_params",
        }
        return pipes.fingerprint(self.dict(exclude=excludes))


SingleSearchFactoryConfig = Union[ElasticsearchFactoryConfig, FaissFactoryConfig, QdrantFactoryConfig]
SingleSearchFactoryDiff = Union[ElasticsearchFactoryDiff, FaissFactoryDiff, QdrantFactoryDiff]

FactoryConfigsByBackend: dict[SearchBackends, Type[BaseSearchFactoryConfig]] = {
    "elasticsearch": ElasticsearchFactoryConfig,
    "faiss": FaissFactoryConfig,
    "qdrant": QdrantFactoryConfig,
}

FactoryDiffByBackend: dict[SearchBackends, Type[BaseSearchFactoryDiff]] = {
    "elasticsearch": ElasticsearchFactoryDiff,
    "faiss": FaissFactoryDiff,
    "qdrant": QdrantFactoryDiff,
}


class MutliSearchFactoryDiff(BaseSearchFactoryDiff):
    """Configures a hybrid search engine."""

    backend: Literal["multi"] = "multi"
    engines: dict[str, SingleSearchFactoryDiff]

    @classmethod
    def parse(cls: Type[Self], config: dict | omegaconf.DictConfig) -> Self:  # type: ignore
        """Parse a dictionary / omegaconf configuration into a structured dict."""
        if "engines" in config:
            config = config["engines"]
        config = _parse_multi_search(config, FactoryDiffByBackend)

        if len(config) == 0:
            raise ValueError(f"Attempting to initialize a `{cls.__name__}` without engines.")

        return cls(engines=config)  # type: ignore


class MutliSearchFactoryConfig(BaseSearchFactoryConfig):
    """Configures a hybrid search engine."""

    _defaults = pydantic.PrivateAttr(None)

    backend: Literal["multi"] = "multi"
    engines: dict[str, SingleSearchFactoryConfig]

    @classmethod
    def parse(cls: Type[Self], config: dict | omegaconf.DictConfig) -> Self:  # type: ignore
        """Parse a dictionary / omegaconf configuration into a structured dict."""
        if "engines" in config:
            config = config["engines"]
        config = _parse_multi_search(config, FactoryConfigsByBackend)

        if len(config) == 0:
            raise ValueError(f"Attempting to initialize a `{cls.__name__}` without engines.")

        return cls(engines=config)  # type: ignore

    def __add__(self, diff: None | MutliSearchFactoryDiff) -> Self:
        if diff is None:
            return self
        new_engines = copy.copy(self.engines)
        for key, engine in diff.engines.items():
            if self.engines[key].backend == engine.backend:
                new_engines[key] = self.engines[key] + engine  # type: ignore
            elif self._defaults is not None:
                default_engine = getattr(self._defaults, engine.backend)
                new_engines[key] = default_engine + engine
            else:
                raise ValueError("`_defaults` was never set.")

        return self.copy(update={"engines": new_engines})


class SearchFactoryDefaults(StrictModel):
    """Default configurations for the search engine backend."""

    elasticsearch: ElasticsearchFactoryConfig = ElasticsearchFactoryConfig()
    faiss: FaissFactoryConfig = FaissFactoryConfig()
    qdrant: QdrantFactoryConfig = QdrantFactoryConfig()

    # validators
    _validate_elasticsearch = pydantic.validator("elasticsearch", allow_reuse=True, pre=True)(as_pyobj_validator)
    _validate_faiss = pydantic.validator("faiss", allow_reuse=True, pre=True)(as_pyobj_validator)
    _validate_qdrant = pydantic.validator("qdrant", allow_reuse=True, pre=True)(as_pyobj_validator)

    @classmethod
    def parse(cls: Type[Self], config: dict | omegaconf.DictConfig) -> Self:  # type: ignore
        """Parse a dictionary / omegaconf configuration into a structured dict."""
        return cls(**config)

    # methods
    def __add__(self, diff: MutliSearchFactoryDiff) -> MutliSearchFactoryConfig:
        engine_factories = {}
        for key, cfg_diff in diff.engines.items():
            default_config = getattr(self, cfg_diff.backend)
            engine_factories[key] = default_config + cfg_diff

        cfg = MutliSearchFactoryConfig(engines=engine_factories)
        cfg._defaults = self  # <- save the defaults for better resolution of diffs
        return cfg


T = TypeVar("T")
K = TypeVar("K")


def _parse_multi_search(
    config: dict | omegaconf.DictConfig,  # type: ignore
    sub_cls_by_backend: dict[K, Type[T]],
) -> dict[K, T]:  # type: ignore
    """Parse a dictionary / omegaconf configuration into a structured dict."""
    if isinstance(config, omegaconf.DictConfig):
        config: dict = omegaconf.OmegaConf.to_container(config, resolve=True)  # type: ignore

    formatted_config = {}
    for engine_name, cfg in config.items():
        try:
            backend = cfg["backend"]
        except KeyError as exc:
            raise KeyError(
                f"Backend must be configured. Found configuration keys `{list(cfg.keys())}`. Missing=`backend`"
            ) from exc
        try:
            sub_cls = sub_cls_by_backend[backend]
        except KeyError as exc:
            raise KeyError(f"Unknown backend `{backend}`. Known backends: {list(sub_cls_by_backend.keys())}") from exc

        # Instantiate the specific engine
        formatted_config[engine_name] = sub_cls(**cfg)

    return formatted_config
