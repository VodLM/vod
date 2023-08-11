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
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger
from torch.utils import data as torch_data
from typing_extensions import Self, Type
from vod_tools import dstruct
from vod_tools.misc.schedule import BaseSchedule

from src import vod_configs, vod_dataloaders, vod_search

T = TypeVar("T")
K = TypeVar("K")


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


@dataclasses.dataclass(frozen=True)
class PrecomputedDsetVectors:
    """Holds the vectors for a given dataset and field."""

    questions: dstruct.TensorStoreFactory
    sections: dstruct.TensorStoreFactory


@dataclasses.dataclass(frozen=True)
class DsetWithVectors:
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
    def cast(
        cls: Type[Self],
        *,
        data: dstruct.SizedDataset[dict[str, Any]]
        | datasets.Dataset
        | list[datasets.Dataset]
        | list[dstruct.SizedDataset[dict[str, Any]]],
        vectors: None
        | np.ndarray
        | dstruct.TensorStoreFactory
        | dstruct.SizedDataset[np.ndarray]
        | list[dstruct.TensorStoreFactory]
        | list[dstruct.SizedDataset[np.ndarray]]
        | list[np.ndarray] = None,
    ) -> Self:
        """Cast a dataset and vectors to the correct type."""
        if isinstance(data, datasets.Dataset):
            pass
        elif isinstance(data, list) and all(isinstance(d, datasets.Dataset) for d in data):
            data = datasets.concatenate_datasets(data)  # type: ignore
        else:
            dstruct.ConcatenatedSizedDataset(data)  # type: ignore

        if vectors is not None:
            if isinstance(vectors, list):
                vectors = dstruct.ConcatenatedSizedDataset([dstruct.as_lazy_array(v) for v in vectors])
            else:
                vectors = dstruct.as_lazy_array(vectors)

        return cls(
            data=data,  # type: ignore
            vectors=vectors,
        )

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return (
            f"{type(self).__name__}(data.len={len(self.data)}, "
            f"vectors.len={len(self.vectors) if self.vectors is not None else None})"
        )


def instantiate_retrieval_dataloader(
    *,
    queries: DsetWithVectors,
    sections: DsetWithVectors,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    search_client: vod_search.ShardedMultiSearchClient,
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


D = TypeVar("D", bound=typing.Union[dstruct.SizedDataset, datasets.Dataset])


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


@dataclasses.dataclass(frozen=False)
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
