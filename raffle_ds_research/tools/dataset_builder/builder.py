from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Generic, List, Optional, Protocol, Type, TypeVar, Union

import datasets
import numpy as np
import omegaconf
import pydantic
import torch
from pydantic import BaseModel, ValidationError


class DatasetProtocol(Protocol):
    def __getitem__(self, index: int) -> dict:
        ...

    def __len__(self) -> int:
        ...


class CollateFnProtocol(Protocol):
    def __call__(self, batch: List[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        ...


D = TypeVar("D", bound=DatasetProtocol)
DD = TypeVar("DD", bound=Union[dict[str, DatasetProtocol], datasets.DatasetDict])
DorDD = TypeVar("DorDD", bound=Union[DatasetProtocol, dict[str, DatasetProtocol], datasets.DatasetDict])
LoadCfg = TypeVar("LoadCfg", bound=pydantic.BaseModel)


class DatasetBuilder(Generic[DD, LoadCfg], ABC):
    """Abstract object for all dataset datasets."""

    _loader_config: Type[LoadCfg]

    @abstractmethod
    def __call__(self) -> DD:
        """Instantiate dataset and return it."""
        ...

    def get_corpus(self) -> Optional[datasets.Dataset]:
        """Return corpus dataset if it exists."""
        return None

    @staticmethod
    @abstractmethod
    def get_collate_fn(config: Optional[LoadCfg] = None) -> CollateFnProtocol:
        """Return collate function for that dataset."""
        return torch.utils.data.dataloader.default_collate

    @property
    def loader_config(cls) -> Type[LoadCfg]:
        return cls._loader_config


class ExampleValidationError(ValidationError):
    ...


class CollateRuntimeError(RuntimeError):
    ...


class HfBuilder(DatasetBuilder[datasets.DatasetDict, LoadCfg]):
    _loader_config: Type[LoadCfg]
    row_model: Optional[Type[BaseModel]]
    batch_model: Optional[Type[BaseModel]]
    _validate_only_splits: Optional[list[str]]
    _subset_size: Optional[dict[str, int]]
    _hf_load_kwargs: dict[str, Any]
    _prep_map_kwargs: dict[str, Any]

    # TODO: HERE - refactoring builders

    def __init__(
        self,
        name: str,
        subset_name: str,
        row_model: Optional[type(BaseModel)] = None,
        batch_model: Optional[type(BaseModel)] = None,
        hf_load_kwargs: Optional[dict[str, Any]] = None,
        subset_size: Optional[int | dict[str, int]] = None,
        validate_only_splits: Optional[list[str]] = None,
        prep_map_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.subset_name = subset_name
        self.row_model = row_model
        self.batch_model = batch_model
        self._hf_load_kwargs = hf_load_kwargs or {}
        self._subset_size = subset_size
        self._validate_only_splits = validate_only_splits

        if prep_map_kwargs is None:
            prep_map_kwargs = dict(num_proc=4)
        elif isinstance(prep_map_kwargs, omegaconf.DictConfig):
            prep_map_kwargs = omegaconf.OmegaConf.to_container(prep_map_kwargs)
        prep_map_kwargs.update(dict(batched=True, with_indices=True))
        self._prep_map_kwargs = prep_map_kwargs

    def prep_map_kwargs(self, **overrides) -> dict[str, Any]:
        return {**self._prep_map_kwargs, **overrides}

    def __call__(self):
        corpus = self.get_corpus()
        dset = self._build_dset(corpus)
        dset = self._take_subset(dset)
        dset = self._add_row_idx(dset)
        for key, dset_split in dset.items():
            if self._validate_only_splits is None or key in self._validate_only_splits:
                self._validate(dset_split)
        return dset

    def _default_load_config(self) -> LoadCfg:
        return self._loader_config()

    def _build_dset(self, corpus: Optional[datasets.Dataset]) -> datasets.DatasetDict:
        dset = datasets.load_dataset(
            path=self.name,
            name=self.subset_name,
            **self._hf_load_kwargs,
        )
        return dset

    def _validate(self, dset: datasets.Dataset):
        # validate the first row
        if self.row_model is not None:
            row = dset[0]
            self.row_model(**row)

        # collate_fn
        cfg = self._default_load_config()
        xs = [dset[i] for i in range(min(3, len(dset)))]
        collate_fn = self.get_collate_fn(config=cfg)
        try:
            batch = collate_fn(xs)
            if self.batch_model is not None:
                self.batch_model(**batch)
        except Exception as e:
            raise CollateRuntimeError(
                f"Collate function failed on {type(self).__name__} dataset at runtime (validation step). "
                f"Please check the collate function for this dataset."
            ) from e

    @staticmethod
    def get_collate_fn(config: Optional[LoadCfg] = None):
        return torch.utils.data.dataloader.default_collate

    def _take_subset(self, dset: datasets.DatasetDict) -> datasets.DatasetDict:
        """Take a subset of the dataset."""
        if self._subset_size is None:
            return dset

        rgn = np.random.RandomState(0)

        # define the subset size for each split
        subset_size = copy(self._subset_size)
        if isinstance(subset_size, int):
            subset_size = {split: subset_size for split in dset.keys()}
        elif isinstance(subset_size, dict):
            assert set(subset_size.keys()) == set(dset.keys())
        else:
            raise TypeError(f"subset_size must be an int or a dict, not {type(self._subset_size)}")

        # sample the subsets
        new_mapped_qa = {}
        for split, size in subset_size.items():
            ids = rgn.choice(list(range(len(dset[split]))), size=size, replace=False)
            new_mapped_qa[split] = dset[split].select(ids)

        return datasets.DatasetDict(new_mapped_qa)

    def _add_row_idx(self, dset: DorDD) -> DorDD:
        """Add row index to each row."""
        prep_map_kwargs = self.prep_map_kwargs(batched=False, desc="Adding row indices")
        dset = dset.map(_add_row_idx, **prep_map_kwargs)
        return dset


ROW_IDX_COL_NAME: str = "__row_idx__"


def _add_row_idx(eg: dict, idx: int) -> dict[str, int]:
    return {ROW_IDX_COL_NAME: idx}
