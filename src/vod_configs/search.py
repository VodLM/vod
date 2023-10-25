import copy
import typing as typ

import faiss
import omegaconf
import pydantic
import torch
from loguru import logger
from typing_extensions import Self, Type
from vod_configs.utils.base import StrictModel
from vod_tools import fingerprint, pretty
from vod_tools.misc.config import as_pyobj_validator

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


def _get_gpu_resources(devices: list[int], tempmem: int = -1, log_mem_allocation: bool = False) -> list[GpuResources]:  # type: ignore
    """Return a list of GPU resources."""
    gpu_resources = []
    ngpu = torch.cuda.device_count() if devices is None else len(devices)
    for i in range(ngpu):
        res = StandardGpuResources()  # type: ignore
        res.setLogMemoryAllocations(log_mem_allocation)
        if tempmem is not None and tempmem > 0:
            logger.debug(f"Setting GPU:{i} temporary memory to {pretty.human_format_bytes(tempmem, 'MB')}")
            res.setTempMemory(tempmem)

        gpu_resources.append(res)

    return gpu_resources


class FaissGpuConfig(StrictModel):
    """Configuration for training a faiss index on GPUs."""

    devices: list[int] = [-1]
    use_float16: bool = True
    use_precomputed_tables: bool = True
    max_add: None | int = 2**18
    tempmem: None | int = -1
    keep_indices_on_cpu: bool = False
    verbose: bool = True
    shard: bool = True
    add_batch_size: int = 2**18

    @pydantic.field_validator("devices", mode="before")
    @classmethod
    def _validate_devices(cls: Type[Self], v: None | list[int]) -> list[int]:  # noqa: ANN202
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


SearchBackend = typ.Literal["elasticsearch", "faiss", "qdrant"]


class BaseSearchFactoryConfig(StrictModel):
    """Base config for all search engines."""

    backend: SearchBackend = pydantic.Field(..., description="Search backend to use.")
    subset_id_key: None | str = pydantic.Field(default="subset_id", description="Subset ID field to be indexed.")
    section_id_key: None | str = pydantic.Field(default="id", description="Section ID field to be indexed.")


class BaseSearchFactoryDiff(StrictModel):
    """Relative search configs."""

    backend: SearchBackend
    group_key: None | str = None
    section_id_key: None | str = None


class FaissFactoryDiff(BaseSearchFactoryDiff):
    """Configures a relative faiss configuration."""

    backend: typ.Literal["faiss"] = "faiss"
    factory: None | str = None
    nprobe: None | int = None
    metric: None | int = None
    train_size: None | int = None
    logging_level: None | str = None
    host: None | str = None
    port: None | int = None
    gpu: None | FaissGpuConfig = None


class FaissFactoryConfig(BaseSearchFactoryConfig):
    """Configures the building of a faiss server."""

    backend: typ.Literal["faiss"] = "faiss"
    factory: str = "Flat"
    nprobe: int = 16
    metric: int = faiss.METRIC_INNER_PRODUCT
    train_size: None | int = None
    logging_level: str = "DEBUG"
    host: str = "http://localhost"
    port: int = -1
    gpu: None | FaissGpuConfig = None

    def __add__(self, diff: None | FaissFactoryDiff) -> Self:
        if diff is None:
            return self
        diffs = {k: v for k, v in diff if v is not None}
        return self.model_copy(update=diffs)

    @pydantic.field_validator("metric", mode="before")
    @classmethod
    def _validate_metric(cls: Type[Self], v: str | int) -> int:
        if isinstance(v, int):
            return v

        return FAISS_METRICS[v]

    def fingerprint(self) -> str:
        """Return a fingerprint for this config."""
        excludes = {"host", "port", "logging_level"}
        return fingerprint.fingerprint(self.model_dump(exclude=excludes))


class ElasticsearchFactoryDiff(BaseSearchFactoryDiff):
    """Configures a relative elasticsearch configuration."""

    backend: typ.Literal["elasticsearch"] = "elasticsearch"
    host: None | str = None
    port: None | int = None
    persistent: None | bool = None
    es_body: None | dict = None
    language: None | str = None


class ElasticsearchFactoryConfig(BaseSearchFactoryConfig):
    """Configures the building of an Elasticsearch server."""

    backend: typ.Literal["elasticsearch"] = "elasticsearch"
    host: str = "http://localhost"
    port: int = 9200
    persistent: bool = True
    section_template: str = r"{% if title %}{{ title }}{% endif %} {{ content }}"
    es_body: None | dict = None
    language: None | str = None

    def __add__(self, diff: None | ElasticsearchFactoryDiff) -> Self:
        if diff is None:
            return self
        diffs = {k: v for k, v in diff if v is not None}
        return self.model_copy(update=diffs)

    @pydantic.field_validator("es_body", mode="before")
    @classmethod
    def _validate_es_body(cls: Type[Self], v: dict | None) -> dict | None:
        if isinstance(v, omegaconf.DictConfig):
            v = omegaconf.OmegaConf.to_container(v, resolve=True)  # type: ignore

        return v

    def fingerprint(self, exclude: None | list[str] = None) -> str:  # noqa: ARG002
        """Return a fingerprint for this config."""
        base_exclude = {"host", "port", "persistent"}
        if exclude:
            base_exclude.update(exclude)
        return fingerprint.fingerprint(self.model_dump(exclude=base_exclude))


class QdrantFactoryDiff(BaseSearchFactoryDiff):
    """Configures a relative qdrant configuration."""

    backend: typ.Literal["qdrant"] = "qdrant"
    host: None | str = None
    port: None | int = None
    grpc_port: None | int = None
    persistent: None | bool = None
    exist_ok: None | bool = None
    qdrant_body: None | dict = None
    search_params: None | dict = None
    force_single_collection: None | bool = None


class QdrantFactoryConfig(BaseSearchFactoryConfig):
    """Configures the building of a Qdrant server."""

    backend: typ.Literal["qdrant"] = "qdrant"
    host: str = "http://localhost"
    port: int = 6333
    grpc_port: None | int = 6334
    persistent: bool = False
    exist_ok: bool = True
    qdrant_body: None | dict = None
    search_params: None | dict = None
    force_single_collection: bool = False

    def __add__(self, diff: None | QdrantFactoryDiff) -> Self:
        if diff is None:
            return self
        diffs = {k: v for k, v in diff if v is not None}
        if "qdrant_body" in diffs:
            diffs["qdrant_body"] = self.model_dump()["qdrant_body"] | diffs["qdrant_body"]
        return self.model_copy(update=diffs)

    @pydantic.field_validator("qdrant_body", mode="before")
    @classmethod
    def _validate_qdrant_body(cls: Type[Self], v: dict | None) -> dict | None:
        if isinstance(v, omegaconf.DictConfig):
            v = omegaconf.OmegaConf.to_container(v, resolve=True)  # type: ignore
        return v

    @pydantic.field_validator("search_params", mode="before")
    @classmethod
    def _validate_search_params(cls: Type[Self], v: dict | None) -> dict | None:
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
        return fingerprint.fingerprint(self.model_dump(exclude=excludes))


SingleSearchFactoryConfig = ElasticsearchFactoryConfig | FaissFactoryConfig | QdrantFactoryConfig
SingleSearchFactoryDiff = ElasticsearchFactoryDiff | FaissFactoryDiff | QdrantFactoryDiff

_FactoryConfigsByBackend: dict[SearchBackend, Type[BaseSearchFactoryConfig]] = {
    "elasticsearch": ElasticsearchFactoryConfig,
    "faiss": FaissFactoryConfig,
    "qdrant": QdrantFactoryConfig,
}

_FactoryDiffByBackend: dict[SearchBackend, Type[BaseSearchFactoryDiff]] = {
    "elasticsearch": ElasticsearchFactoryDiff,
    "faiss": FaissFactoryDiff,
    "qdrant": QdrantFactoryDiff,
}


class HybridSearchFactoryDiff(BaseSearchFactoryDiff):
    """Configures a hybrid search engine."""

    backend: typ.Literal["hybrid"] = "hybrid"
    engines: dict[str, SingleSearchFactoryDiff]


class HybridSearchFactoryConfig(BaseSearchFactoryConfig):
    """Configures a hybrid search engine."""

    _defaults = pydantic.PrivateAttr(None)

    backend: typ.Literal["hybrid"] = "hybrid"
    engines: dict[str, SingleSearchFactoryConfig]

    def __add__(self, diff: None | HybridSearchFactoryDiff) -> Self:
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

        return self.model_copy(update={"engines": new_engines})


class SearchFactoryDefaults(StrictModel):
    """Default configurations for the search engine backend."""

    elasticsearch: ElasticsearchFactoryConfig = pydantic.Field(
        default_factory=lambda: ElasticsearchFactoryConfig(),
    )
    faiss: FaissFactoryConfig = pydantic.Field(
        default_factory=lambda: FaissFactoryConfig(),
    )
    qdrant: QdrantFactoryConfig = pydantic.Field(
        default_factory=lambda: QdrantFactoryConfig(),
    )

    # validators
    _validate_elasticsearch = pydantic.field_validator("elasticsearch", mode="before")(as_pyobj_validator)
    _validate_faiss = pydantic.field_validator("faiss", mode="before")(as_pyobj_validator)
    _validate_qdrant = pydantic.field_validator("qdrant", mode="before")(as_pyobj_validator)

    # methods
    def __add__(self, diff: HybridSearchFactoryDiff) -> HybridSearchFactoryConfig:
        engine_factories = {}
        for key, cfg_diff in diff.engines.items():
            default_config = getattr(self, cfg_diff.backend)
            engine_factories[key] = default_config + cfg_diff

        cfg = HybridSearchFactoryConfig(engines=engine_factories)
        cfg._defaults = self  # <- save the defaults for better resolution of diffs
        return cfg
