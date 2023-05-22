from __future__ import annotations

import collections
import dataclasses
import functools
import typing
from typing import Any, Callable, Optional, TypeVar

import datasets
import lightning as L  # noqa: N812
import numpy as np
import rich
import transformers
from loguru import logger
from torch.utils import data as torch_data
from typing_extensions import Self, Type

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core import mechanics
from raffle_ds_research.core.workflows.utils import support
from raffle_ds_research.tools import dstruct, index_tools, pipes

T = TypeVar("T")


def none_ok(func: Callable[..., T]) -> Callable[..., Optional[T]]:
    """Decorator that allows `None` as an input."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
        if args[0] is None:
            return None
        return func(*args, **kwargs)

    return wrapper


maybe_as_lazy_array = none_ok(dstruct.as_lazy_array)


def is_engine_enabled(parameters: Optional[dict[str, float]], engine: str) -> bool:
    """Check if an engine is enabled."""
    if parameters is None:
        return True
    return parameters.get(engine, 1.0) >= 0


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
        data: dstruct.SizedDataset[dict[str, Any]] | datasets.Dataset,
        vectors: None | np.ndarray | dstruct.TensorStoreFactory | np.ndarray | dstruct.SizedDataset[np.ndarray],
    ) -> Self:
        """Cast a dataset and vectors to the correct type."""
        return cls(
            data=data,  # type: ignore
            vectors=dstruct.as_lazy_array(vectors) if vectors is not None else None,
        )

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return (
            f"{type(self).__name__}(data.len={len(self.data)}, "
            f"vectors.len={len(self.vectors) if self.vectors is not None else None})"
        )


def instantiate_retrieval_dataloader(
    *,
    questions: DsetWithVectors,
    sections: DsetWithVectors,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    search_client: index_tools.MultiSearchClient,
    collate_config: core_config.RetrievalCollateConfig,
    dataloader_config: core_config.DataLoaderConfig,
    parameters: Optional[dict[str, float]],
) -> torch_data.DataLoader[dict[str, Any]]:
    """Instantiate a dataloader for the retrieval task."""
    collate_fn = mechanics.RetrievalCollate(
        tokenizer=tokenizer,
        sections=sections.data,
        search_client=search_client,
        config=collate_config,
        parameters=parameters,
    )
    dataset = WithVectors(
        dataset=questions.data,
        vectors=questions.vectors,
        vector_key="vector",
    )
    return torch_data.DataLoader(
        dataset=dataset,  # type: ignore
        collate_fn=collate_fn,
        **dataloader_config.dict(),
    )


class WithVectors(dstruct.SizedDataset[dict[str, Any]]):
    """A wrapper around a dataset that adds vectors to each item."""

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


def concatenate_datasets(dsets: typing.Iterable[support.DsetWithVectors]) -> support.DsetWithVectors:  # TODO: test this
    """Concatenate datasets and remove duplicates."""
    dsets_by_fingerprint = collections.defaultdict(list)
    for dset in dsets:
        fgn = pipes.fingerprint(dset.data)
        dsets_by_fingerprint[fgn].append(dset)

    for fgn, dsets in dsets_by_fingerprint.items():
        logger.debug(f"Gathered ({fgn}) : {[str(d) for d in dsets]}")

    unique_dsets = [dsets[0] for dsets in dsets_by_fingerprint.values()]
    return support.DsetWithVectors(
        data=_concat_data([dset.data for dset in unique_dsets]),
        vectors=_concat_data([dset.vectors for dset in unique_dsets if dset.vectors is not None]),
    )


D = TypeVar("D", bound=typing.Union[dstruct.SizedDataset, datasets.Dataset])


def _concat_data(data: list[D]) -> D:
    if all(isinstance(d, datasets.Dataset) for d in data):
        return datasets.concatenate_datasets(data)  # type: ignore

    return dstruct.ConcatenatedSizedDataset(data)  # type: ignore


def _barrier_fn(name: str, trainer: L.Trainer) -> None:
    """Barrier to synchronize all processes."""
    if trainer.world_size == 1:
        rich.print(">> world size is 1, skipping barrier")
        return

    rich.print(
        f"[bold yellow][Rank {(trainer.global_rank + 1)} / {trainer.world_size}][/bold yellow] waiting: `{name}` ..."
    )
    trainer.strategy.barrier(name)
    rich.print(
        f"[bold green][Rank {(trainer.global_rank + 1)} / {trainer.world_size}][/bold green] completed: `{name}` ..."
    )
