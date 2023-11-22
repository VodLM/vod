import typing as typ

import datasets
import numpy as np
import vod_configs
import vod_search
import vod_types as vt
from torch.utils import data as torch_data
from torch.utils.data.dataloader import _worker_init_fn_t
from typing_extensions import Self, Type
from vod_dataloaders.core.utils import VECTOR_KEY
from vod_dataloaders.dl_sampler import DlSamplerFactory
from vod_search import ShardName

from .realm_collate import RealmCollate

T = typ.TypeVar("T")
K = typ.TypeVar("K")
D = typ.TypeVar("D", bound=vt.Sequence)


class RealmDataloader(torch_data.DataLoader[dict[str, typ.Any]]):
    """A subclass of `torch.utils.data.DataLoader` to implement VOD's magic."""

    @classmethod
    def factory(  # noqa: PLR0913, ANN206, D417
        cls: Type[Self],
        *,
        queries: dict[K, tuple[ShardName, vt.DictsSequence]],
        vectors: None | dict[K, vt.Sequence[np.ndarray]] = None,
        # Parameters for the Collate function
        search_client: vod_search.HybridSearchClient,
        collate_config: vod_configs.RealmCollateConfig,
        parameters: None | typ.MutableMapping = None,
        # Base `torch.utils.data.Dataloader` arguments
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: torch_data.Sampler | typ.Iterable | DlSamplerFactory | None = None,
        batch_sampler: torch_data.Sampler[typ.Sequence] | typ.Iterable[typ.Sequence] | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: _worker_init_fn_t | None = None,
        multiprocessing_context=None,  # noqa: ANN001
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        """Instantiate a `RealmDataloader`.

        NOTE: We don't override the `__init__` method because doing so breaks
            the base functionality of `torch.utils.data.DataLoader` (multiprocessing).

        Args:
            queries: the dataset of queries.
            vectors: the vectors associated with the queries.
            search_client: the hybrid search client (e.g. elasticsearch + faiss).
            tokenizer: the huggingface tokenizer.
            collate_config: the configuration for the `RealmCollate` function.
            parameters: the parameters used to weight the search engines (e.g., dense, sparse).

        """
        queries_shards = {s for s, _ in queries.values()}
        if queries_shards != set(search_client.shard_list):
            raise ValueError(
                "The keys (shard) of `queries` and `search_client` must match. "
                f"Found {queries_shards} and {search_client.shard_list}"
            )

        # Validate the vectors
        if vectors is not None and set(vectors.keys()) != set(queries.keys()):
            raise ValueError("The keys (dataset ID) of `queries_vectors` and `queries` must match.")
        vectors = vectors or {}

        # Augment the queries rows with
        #    the shard information (at key `TARGET_SHARD_KEY``)
        #    and cached vectors at key `VECTOR_KEY`.
        queries_with_extras: list[vt.DictsSequence] = [
            _WithExtrasAndVectors(
                dataset=dset,
                vectors=vectors.get(key, None),
                extras={vod_configs.TARGET_SHARD_KEY: shard},
            )
            for key, (shard, dset) in queries.items()
        ]

        # Concatenate the queries into a single sequence
        concatenated_queries = _concatenate_dsets(queries_with_extras)

        # Instantiate the Collate function
        collate_fn = RealmCollate(
            search_client=search_client,
            config=collate_config,
            parameters=parameters,
        )

        # Instantiate the Sampler
        if isinstance(sampler, DlSamplerFactory):
            sampler = sampler(concatenated_queries)

        return cls(
            dataset=concatenated_queries,  # type: ignore
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,  # type: ignore
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )


class _WithExtrasAndVectors(vt.DictsSequence[T]):
    """A wrapper around a dataset to add extra fields and vectors to a sampled row."""

    __slots__ = ("dataset", "vectors", "vector_key", "extras")

    def __init__(
        self,
        *,
        dataset: vt.DictsSequence,
        vectors: None | vt.Sequence[np.ndarray],
        vector_key: str = VECTOR_KEY,
        extras: dict[str, T],
    ) -> None:
        self.dataset = dataset
        self.vectors = vectors
        self.vector_key = vector_key
        self.extras = extras or {}

    def __getitem__(self, index: int) -> dict[str, T]:
        row = self.dataset[index]
        row.update(self.extras)
        if self.vectors is not None:
            row[self.vector_key] = self.vectors[index]
        return row

    def __len__(self) -> int:
        return len(self.dataset)

    def _vector_desc(self) -> str:
        if self.vectors is None:
            return "None"
        dims = [len(self.vectors), *self.vectors[0].shape]
        return f"[{','.join([str(d) for d in dims])}]"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vectors={self._vector_desc()}, extras={self.extras}, dataset={self.dataset})"


def _concatenate_dsets(parts: list[D]) -> D:
    """Concatenate a list of datasets."""
    if len(parts) > 1:
        if all(isinstance(p, datasets.Dataset) for p in parts):
            return datasets.concatenate_datasets(parts)  # type: ignore
        raise NotImplementedError(
            f"Concatenation is only supported for type `datasets.Dataset`. Found types {[type(p) for p in parts]}"
        )

    return parts[0]
