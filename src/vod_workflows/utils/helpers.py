import contextlib
import dataclasses
import functools
import typing as typ

import lightning as L
import numpy as np
import torch
import vod_configs
import vod_datasets
import vod_types as vt
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger
from typing_extensions import Self, Type
from vod_search.base import ShardName
from vod_tools.misc.schedule import BaseSchedule, schedule_factory
from vod_tools.ts_factory.ts_factory import TensorStoreFactory

T = typ.TypeVar("T")
K = typ.TypeVar("K")
D = typ.TypeVar("D", bound=vt.Sequence)


@dataclasses.dataclass
class RetrievalTask(typ.Generic[K]):
    """A retrieval task with queries, sections and the config required to build a search engine."""

    queries: list[vod_configs.QueriesDatasetConfig]
    sections: list[vod_configs.SectionsDatasetConfig]
    vectors: None | dict[vod_configs.BaseDatasetConfig, TensorStoreFactory]


def none_ok(func: typ.Callable[..., T]) -> typ.Callable[..., None | T]:
    """Decorator that allows `None` as an input."""

    @functools.wraps(func)
    def wrapper(*args: typ.Any, **kwargs: typ.Any) -> None | T:
        if args[0] is None:
            return None
        return func(*args, **kwargs)

    return wrapper


maybe_as_lazy_array = none_ok(vt.as_lazy_array)


def is_engine_enabled(parameters: None | typ.MutableMapping, engine: str) -> bool:
    """Check if an engine is enabled."""
    if parameters is None:
        return True
    return parameters.get(engine, 1.0) >= 0


@dataclasses.dataclass(frozen=True)
class QueriesWithVectors:
    """Holds a dict of queries and their vectors."""

    queries: dict[str, tuple[ShardName, vt.DictsSequence]]
    vectors: None | dict[str, vt.Sequence[np.ndarray]]

    @classmethod
    def from_configs(
        cls: Type[Self],
        queries: list[vod_configs.QueriesDatasetConfig],
        vectors: None | dict[vod_configs.BaseDatasetConfig, TensorStoreFactory],
    ) -> Self:
        """Load a list of datasets from their respective configs."""
        key_map = {cfg.hexdigest(): cfg for cfg in queries}
        queries_by_key = {key: (cfg.link, vod_datasets.load_queries(cfg)) for key, cfg in key_map.items()}
        vectors_by_key = (
            {key: vt.as_lazy_array(vectors[cfg]) for key, cfg in key_map.items()} if vectors is not None else None
        )
        return cls(
            queries=queries_by_key,  # type: ignore
            vectors=vectors_by_key,  # type: ignore
        )


@dataclasses.dataclass(frozen=True)
class SectionsWithVectors(typ.Generic[K]):
    """Holds a dict of sections and their vectors."""

    sections: dict[ShardName, vt.DictsSequence]
    vectors: None | dict[ShardName, vt.Sequence[np.ndarray]]
    search_configs: dict[ShardName, vod_configs.HybridSearchFactoryConfig]

    @classmethod
    def from_configs(
        cls: Type[Self],
        sections: list[vod_configs.SectionsDatasetConfig],
        vectors: None | dict[vod_configs.BaseDatasetConfig, TensorStoreFactory],
    ) -> Self:
        """Load a list of datasets from their respective configs."""
        sections_by_shard_name = {cfg.identifier: vod_datasets.load_sections(cfg) for cfg in sections}
        vectors_by_shard_name = (
            {cfg.identifier: vt.as_lazy_array(vectors[cfg]) for cfg in sections} if vectors is not None else None
        )
        configs_by_shard_name = {cfg.identifier: cfg.search for cfg in sections}
        return cls(
            sections=sections_by_shard_name,  # type: ignore
            vectors=vectors_by_shard_name,  # type: ignore
            search_configs=configs_by_shard_name,  # type: ignore
        )


def barrier_fn(name: str, fabric: L.Fabric) -> None:
    """Barrier to synchronize all processes."""
    if fabric.world_size == 1:
        return
    with contextlib.suppress(TypeError):
        logger.level("WAIT", no=12, color="<magenta>", icon="⏳")
        logger.level("PASS", no=13, color="<cyan>", icon="✅")
    logger.log("WAIT", f"barrier:wait: `{name}`")
    fabric.strategy.barrier(name)
    logger.log("PASS", f"barrier:pass: `{name}`")


@dataclasses.dataclass
class TrainerState:
    """Holds the state of the trainer."""

    step: int
    epoch: int
    pidx: int
    period: int | list[int]
    period_max_steps: None | int
    max_steps: int
    parameters: dict[str, BaseSchedule] = dataclasses.field(default_factory=dict)
    val_check_interval: int = 500
    log_interval: int = 100
    accumulate_grad_batches: int = 1
    gradient_clip_val: None | float = None
    n_max_eval: None | int = None

    def get_parameters(self) -> dict[str, float]:
        """Return the parameters for a given step."""
        return {k: v(self.step) for k, v in self.parameters.items()}

    def __getstate__(self) -> dict[str, typ.Any]:
        """Return the state of the object."""
        state = dataclasses.asdict(self)
        state["parameters"] = {k: v.model_dump() for k, v in self.parameters.items()}
        return state

    def __setstate__(self, state: dict[str, typ.Any]) -> None:
        """Set the state of the object."""
        state["parameters"] = {k: schedule_factory(**v) for k, v in state["parameters"].items()}
        self.__dict__.update(state)


class _OptimizerWrapper(typ.Protocol):
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        ...


def unwrap_optimizer(optimizer: torch.optim.Optimizer | _OptimizerWrapper) -> torch.optim.Optimizer:
    """Unwrap the optimizer if it is wrapped."""
    while True:
        if isinstance(optimizer, torch.optim.Optimizer) and not type(optimizer).__name__.startswith("Fabric"):
            break
        try:
            optimizer = optimizer.optimizer  # type: ignore
        except AttributeError as exc:
            raise AttributeError(f"Could not find optimizer in `{optimizer}`") from exc

    return optimizer


unwrap_fabric_object = fabric_wrappers._unwrap_objects


def _gen_dummy_batch(bs: int = 8, r: int = 0) -> dict[str, torch.Tensor]:
    return {
        "question.input_ids": r + torch.randint(0, 100, (bs, 10)),
        "question.attention_mask": torch.ones((bs, 10), dtype=torch.long),
        "section.input_ids": r + torch.randint(0, 100, (bs, 8, 10)),
        "section.attention_mask": torch.ones((bs, 8, 10), dtype=torch.long),
    }
