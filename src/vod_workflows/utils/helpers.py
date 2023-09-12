from __future__ import annotations

import contextlib
import dataclasses
import functools
import typing
from multiprocessing.managers import DictProxy
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar

import datasets  # noqa: E402
import lightning as L
import numpy as np
import torch
import transformers
import vod_datasets
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger
from torch.utils import data as torch_data
from typing_extensions import Self, Type
from vod_tools import dstruct
from vod_tools.misc.schedule import BaseSchedule, schedule_factory

from src import vod_configs, vod_dataloaders, vod_search

T = TypeVar("T")
K = TypeVar("K")
D = TypeVar("D", bound=typing.Union[dstruct.SizedDataset, datasets.Dataset])


def _maybe_concatenate_parts(parts: list[D]) -> D:
    if len(parts) > 1:
        if all(isinstance(p, datasets.Dataset) for p in parts):
            data_ = datasets.concatenate_datasets(parts)  # type: ignore
        else:
            data_ = dstruct.ConcatenatedSizedDataset(parts)  # type: ignore
    else:
        data_ = parts[0]

    return data_  # type: ignore


@dataclasses.dataclass
class RetrievalTask(Generic[K]):
    """A retrieval task with queries, sections and the config required to build a search engine."""

    queries: list[vod_configs.QueriesDatasetConfig]
    sections: list[vod_configs.SectionsDatasetConfig]
    vectors: None | dict[vod_configs.BaseDatasetConfig, dstruct.TensorStoreFactory]


def none_ok(func: Callable[..., T]) -> Callable[..., Optional[T]]:
    """Decorator that allows `None` as an input."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
        if args[0] is None:
            return None
        return func(*args, **kwargs)

    return wrapper


maybe_as_lazy_array = none_ok(dstruct.as_lazy_array)


def is_engine_enabled(parameters: Optional[dict | DictProxy], engine: str) -> bool:
    """Check if an engine is enabled."""
    if parameters is None:
        return True
    return parameters.get(engine, 1.0) >= 0


class _WithExtrasAttributes(dstruct.SizedDataset[dict[str, Any]]):
    """A class that adds extra attributes to the accessed items."""

    __slots__ = ("data", "extras")

    def __init__(
        self,
        *,
        data: dstruct.SizedDataset[dict[str, Any]] | datasets.Dataset,
        extras: dict[str, Any],
    ) -> None:
        self.data = data
        self.extras = extras

    def __len__(self) -> int:
        """Get the length data the dataset."""
        return len(self.data)

    def __getitem__(self, index: int | list[int] | slice) -> dict[str, Any]:
        """Get an item from the dataset and add the extras."""
        item = self.data[index]
        item.update(self.extras)
        return item

    def __repr__(self) -> str:
        return f"{type(self).__name__}(extras={self.extras}, data={self.data})"


@dataclasses.dataclass(frozen=True)
class PrecomputedDsetVectors:
    """Holds the vectors for a given dataset and field."""

    questions: dstruct.TensorStoreFactory
    sections: dstruct.TensorStoreFactory


def _load_dataset_with_target_shard(config: vod_configs.BaseDatasetConfig) -> dstruct.SizedDataset[dict[str, Any]]:
    """Load the dataset and potentially wrap it to include the target shard name."""
    if isinstance(config, vod_configs.SectionsDatasetConfig):
        return vod_datasets.load_sections(config)  # type: ignore
    if isinstance(config, vod_configs.QueriesDatasetConfig):
        data = vod_datasets.load_queries(config)  # type: ignore
        return _WithExtrasAttributes(data=data, extras={vod_configs.TARGET_SHARD_KEY: config.link})

    raise TypeError(f"Unsupported dataset config: `{config}`")


@dataclasses.dataclass(frozen=True)
class ShardedDsetWithVectors:
    """Holds a dataset and its vectors."""

    data: dstruct.SizedDataset[dict[str, Any]]
    vectors: None | dstruct.SizedDataset[np.ndarray]

    def __post_init__(self):
        """Check that the dataset and vectors have the same length."""
        if self.vectors is not None and len(self.data) != len(self.vectors):
            raise ValueError(
                f"Dataset and vectors must have the same length, "
                f"but got {len(self.data)} and {len(self.vectors)}, respectively."
            )

    @classmethod
    def from_configs(
        cls: Type[Self],
        *,
        data: list[vod_configs.QueriesDatasetConfig | vod_configs.SectionsDatasetConfig],
        vectors: None
        | np.ndarray
        | list[dstruct.TensorStoreFactory]
        | list[dstruct.SizedDataset[np.ndarray]]
        | list[np.ndarray] = None,
    ) -> Self:
        """Load multiple dataset shards from their respective configs.

        NB: query dataset are augmented such that each item has a `__LINKED_SHARD__` key.
        This key is necessary for the sharded search engine to function. It would be great
        to find a better way to do this. For instance, implementing a custom `torch.utils.data.DataLoader`,
        which would automatically concatenate datasets, could do the trick.
        """
        if vectors is not None and len(data) != len(vectors):
            raise ValueError(
                f"Dataset and vectors must have the same length, "
                f"but got {len(data)} and {len(vectors)}, respectively."
            )

        # Load the datasets and concatenate them
        data_ = _maybe_concatenate_parts([_load_dataset_with_target_shard(cfg) for cfg in data])

        # Concatenate the vectors
        if vectors is not None:
            vectors_ = _maybe_concatenate_parts([dstruct.as_lazy_array(v) for v in vectors])
        else:
            vectors_ = None

        return cls(
            data=data_,  # type: ignore
            vectors=vectors_,
        )

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return (
            f"{type(self).__name__}(data.len={len(self.data)}, "
            f"vectors.len={len(self.vectors) if self.vectors is not None else None})"
        )


def instantiate_retrieval_dataloader(
    *,
    queries: ShardedDsetWithVectors,
    sections: ShardedDsetWithVectors,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    search_client: vod_search.HybridSearchClient,
    collate_config: vod_configs.RetrievalCollateConfig,
    dataloader_config: vod_configs.DataLoaderConfig,
    parameters: Optional[dict | DictProxy],
    dl_sampler: typing.Optional[vod_dataloaders.SamplerFactory] = None,
) -> torch_data.DataLoader[dict[str, Any]]:
    """Instantiate a dataloader for the retrieval task."""
    collate_fn = vod_dataloaders.RetrievalCollate(
        tokenizer=tokenizer,
        sections=sections.data,
        search_client=search_client,
        config=collate_config,
        parameters=parameters,
    )
    dataset = IndexWithVectors(
        dataset=queries.data,
        vectors=queries.vectors,
        vector_key="vector",
    )
    kws = dataloader_config.dict()
    if dl_sampler is not None:
        kws["sampler"] = dl_sampler(queries.data)
        kws["shuffle"] = False
    return torch_data.DataLoader(dataset=dataset, collate_fn=collate_fn, **kws)  # type: ignore


class IndexWithVectors(dstruct.SizedDataset[dict[str, Any]]):
    """A wrapper around a dataset that adds vectors to each accessed item."""

    __slots__ = ("dataset", "vectors", "vector_key")

    def __init__(
        self,
        *,
        dataset: dstruct.SizedDataset[dict[str, Any]] | datasets.Dataset,
        vectors: None | dstruct.SizedDataset[np.ndarray],
        vector_key: str = "vector",
    ) -> None:
        self.dataset = dataset
        self.vectors = vectors
        self.vector_key = vector_key

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int | list[int] | slice) -> dict[str, Any]:
        """Get an item from the dataset and inhject the vector."""
        item = self.dataset[index]
        if self.vectors is not None:
            item[self.vector_key] = self.vectors[index]
        return item


def _concat_data(data: list[D]) -> D:
    if len(data) == 1:
        return data[0]

    if all(isinstance(d, datasets.Dataset) for d in data):
        return datasets.concatenate_datasets(data)  # type: ignore

    return dstruct.ConcatenatedSizedDataset(data)  # type: ignore


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
    period_max_steps: Optional[int]
    max_steps: int
    parameters: dict[str, BaseSchedule] = dataclasses.field(default_factory=dict)
    val_check_interval: int = 500
    log_interval: int = 100
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None
    n_max_eval: Optional[int] = None

    def get_parameters(self) -> dict[str, float]:
        """Return the parameters for a given step."""
        return {k: v(self.step) for k, v in self.parameters.items()}

    def __getstate__(self) -> dict[str, Any]:
        """Return the state of the object."""
        state = dataclasses.asdict(self)
        state["parameters"] = {k: v.model_dump() for k, v in self.parameters.items()}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the object."""
        state["parameters"] = {k: schedule_factory(**v) for k, v in state["parameters"].items()}
        self.__dict__.update(state)


class _OptimizerWrapper(Protocol):
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        ...


def unwrap_optimizer(optimizer: torch.optim.Optimizer | _OptimizerWrapper) -> torch.optim.Optimizer:
    """Unwrap the optimizer if it is wrapped."""
    while True:
        if isinstance(optimizer, torch.optim.Optimizer) and not type(optimizer).__name__.startswith("Fabric"):
            break
        try:
            optimizer = optimizer.optimizer
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
