import dataclasses
import typing as typ

import numpy as np
import vod_configs
import vod_datasets
import vod_types as vt
from typing_extensions import Self, Type
from vod_search.base import ShardName

T = typ.TypeVar("T")
K = typ.TypeVar("K")


@dataclasses.dataclass(frozen=True)
class QueriesWithVectors:
    """Holds a dict of queries and their vectors."""

    queries: dict[str, tuple[ShardName, vt.DictsSequence]]
    vectors: None | dict[str, vt.Sequence[np.ndarray]]
    descriptor: None | str = None

    @classmethod
    def from_configs(
        cls: Type[Self],
        queries: list[vod_configs.QueriesDatasetConfig],
        vectors: None | dict[vod_configs.DatasetConfig, vt.Array],
    ) -> Self:
        """Load a list of datasets from their respective configs."""
        descriptor = "+".join(sorted(cfg.identifier for cfg in queries))
        key_map = {cfg.hexdigest(): cfg for cfg in queries}
        queries_by_key = {key: (cfg.link, vod_datasets.load_queries(cfg)) for key, cfg in key_map.items()}
        vectors_by_key = (
            {key: vt.as_lazy_array(vectors[cfg]) for key, cfg in key_map.items()} if vectors is not None else None
        )
        return cls(
            descriptor=descriptor,
            queries=queries_by_key,  # type: ignore
            vectors=vectors_by_key,  # type: ignore
        )

    def __repr__(self) -> str:
        vec_dict = {k: _repr_vector_shape(v) for k, v in self.vectors.items()} if self.vectors else None
        return f"{type(self).__name__}(queries={self.queries}, vectors={vec_dict})"


@dataclasses.dataclass(frozen=True)
class SectionsWithVectors(typ.Generic[K]):
    """Holds a dict of sections and their vectors."""

    sections: dict[ShardName, vt.DictsSequence]
    vectors: None | dict[ShardName, vt.Sequence[np.ndarray]]
    search_configs: dict[ShardName, vod_configs.HybridSearchFactoryConfig]
    descriptor: None | str = None

    @classmethod
    def from_configs(
        cls: Type[Self],
        sections: list[vod_configs.SectionsDatasetConfig],
        vectors: None | dict[vod_configs.DatasetConfig, vt.Array],
    ) -> Self:
        """Load a list of datasets from their respective configs."""
        descriptor = "+".join(sorted(cfg.identifier for cfg in sections))
        sections_by_shard_name = {cfg.identifier: vod_datasets.load_sections(cfg) for cfg in sections}
        vectors_by_shard_name = (
            {cfg.identifier: vt.as_lazy_array(vectors[cfg]) for cfg in sections} if vectors is not None else None
        )
        configs_by_shard_name = {cfg.identifier: cfg.search for cfg in sections}
        return cls(
            descriptor=descriptor,
            sections=sections_by_shard_name,  # type: ignore
            vectors=vectors_by_shard_name,  # type: ignore
            search_configs=configs_by_shard_name,  # type: ignore
        )

    def __repr__(self) -> str:
        vec_dict = {k: _repr_vector_shape(v) for k, v in self.vectors.items()} if self.vectors else None
        return f"{type(self).__name__}(sections={self.sections}, vectors={vec_dict})"


def _repr_vector_shape(x: None | vt.Sequence[np.ndarray]) -> str:
    """Return a string representation of the vectors."""
    if x is None:
        return "None"
    dims = [len(x), *x[0].shape]
    return f"[{', '.join(map(str, dims))}]"
