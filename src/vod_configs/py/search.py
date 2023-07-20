from typing import Optional

import faiss
import omegaconf
import pydantic
import torch
from loguru import logger
from typing_extensions import Self, Type
from vod_configs.py.utils import StrictModel

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


class FaissFactoryConfig(StrictModel):
    """Configures the building of a faiss server."""

    factory: str = "Flat"
    nprobe: int = 16
    metric: int = faiss.METRIC_INNER_PRODUCT
    train_size: Optional[int] = None
    logging_level: str = "DEBUG"
    host: str = "http://localhost"
    port: int = 7678
    gpu: Optional[FaissGpuConfig] = None

    @pydantic.validator("metric", pre=True)
    def _validate_metric(cls, v: str | int) -> int:
        if isinstance(v, int):
            return v

        return FAISS_METRICS[v]

    def fingerprint(self) -> str:
        """Return a fingerprint for this config."""
        excludes = {"host", "port", "logging_level"}
        return pipes.fingerprint(self.dict(exclude=excludes))


class ElasticsearchFactoryConfig(StrictModel):
    """Configures the building of an Elasticsearch server."""

    text_key: str = "text"
    group_key: Optional[str] = "group_hash"
    section_id_key: Optional[str] = "id"
    host: str = "http://localhost"
    port: int = 9200
    persistent: bool = True
    es_body: Optional[dict] = None

    @pydantic.validator("es_body", pre=True)
    def _validate_es_body(cls, v: dict | None) -> dict | None:
        if isinstance(v, omegaconf.DictConfig):
            v = omegaconf.OmegaConf.to_container(v, resolve=True)  # type: ignore
        return v

    def fingerprint(self) -> str:
        """Return a fingerprint for this config."""
        excludes = {"host", "port", "persistent"}
        return pipes.fingerprint(self.dict(exclude=excludes))


class QdrantFactoryConfig(StrictModel):
    """Configures the building of a Qdrant server."""

    group_key: Optional[str] = "group_hash"
    host: str = "http://localhost"
    port: int = 6333
    grpc_port: Optional[int] = 6334
    persistent: bool = False
    exist_ok: bool = True
    qdrant_body: Optional[dict] = None
    search_params: Optional[dict] = None
    force_single_collection: bool = False

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
        excludes = {"host", "port", "persistent"}
        return pipes.fingerprint(self.dict(exclude=excludes))


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


class SearchConfig(StrictModel):
    """Base configuration for search engines (sparse (elasticsearch), dense (faiss, qdrant))."""

    text_key: str = "text"
    group_key: str = "group_hash"
    dense: Optional[FaissFactoryConfig | QdrantFactoryConfig] = None
    sparse: Optional[ElasticsearchFactoryConfig] = None

    @pydantic.validator("sparse", pre=True)
    def _validate_sparse(cls, v: dict | None) -> ElasticsearchFactoryConfig | None:
        if v is None or isinstance(v, ElasticsearchFactoryConfig):
            return v

        if isinstance(v, (dict, omegaconf.DictConfig)):
            return ElasticsearchFactoryConfig(**v)

        return v

    @pydantic.validator("dense", pre=True)
    def _validate_dense(cls, v: dict | None) -> FaissFactoryConfig | QdrantFactoryConfig | None:
        if v is None or isinstance(v, (FaissFactoryConfig, QdrantFactoryConfig)):
            return v

        if isinstance(v, omegaconf.DictConfig):
            v = omegaconf.OmegaConf.to_container(v, resolve=True)  # type: ignore

        if not isinstance(v, dict):
            raise ValueError(f"Invalid type for `dense`: {type(v)}")

        try:
            backend = v.pop("backend")
        except KeyError as exc:
            raise ValueError("`backend` must be set for a dense index.") from exc

        if backend == "faiss":
            return FaissFactoryConfig(**v)

        if backend == "qdrant":
            return QdrantFactoryConfig(**v)

        raise ValueError(f"Unknown backend: {backend}")

    @classmethod
    def parse(cls: Type[Self], obj: dict | omegaconf.DictConfig) -> Self:
        """Parse a config object."""
        return cls(**obj)  # type: ignore
