from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from typing import Generic, List, Optional, Protocol, TypeVar, Union

import numpy as np
import torch
from datasets import Dataset as HfDataset
from datasets import DatasetDict as HfDatasetDict
from datasets import load_dataset
from pydantic import BaseModel, ValidationError


class DatasetProtocol(Protocol):
    def __getitem__(self, index: int) -> dict:
        ...

    def __len__(self) -> int:
        ...


class CollateFnProtocol(Protocol):
    def __call__(self, batch: List[dict]) -> dict:
        ...


D = TypeVar("D", bound=DatasetProtocol)
DD = TypeVar("DD", bound=Union[dict[str, DatasetProtocol], HfDatasetDict])


class DatasetBuilder(Generic[DD], ABC):
    """Abstract object for all dataset datasets."""

    @abstractmethod
    def __call__(self) -> DD:
        """Instantiate dataset and return it."""
        ...

    @staticmethod
    @abstractmethod
    def get_collate_fn(split: Optional[str] = None) -> CollateFnProtocol:
        """Return collate function for that dataset."""
        return torch.utils.data.dataloader.default_collate


class ExampleValidationError(ValidationError):
    ...


class CollateRuntimeError(RuntimeError):
    ...


class HfBuilder(DatasetBuilder[HfDatasetDict]):
    validator: Optional[type(BaseModel)] = None
    batch_validator: Optional[type(BaseModel)] = None
    _validate_splits: list[str] = ["train", "validation", "test"]

    def __init__(
        self,
        name: str,
        validator: Optional[type(BaseModel)] = None,
        batch_validator: Optional[type(BaseModel)] = None,
        load_kwargs: Optional[dict] = None,
        subset_size: Optional[int | dict[str, int]] = None,
    ):
        self.name = name
        self.validator = validator
        self.batch_validator = batch_validator
        self.load_kwargs = load_kwargs or {}
        self.subset_size = subset_size

    def __call__(self):
        dset = self._build_dset()
        for key, dset_split in dset.items():
            if key in self._validate_splits:
                self._validate(dset_split)
        return dset

    def _build_dset(self) -> HfDatasetDict:
        dset = load_dataset(self.name, **self.load_kwargs)
        return dset

    def _validate(self, dset: HfDataset):
        # validate the first row
        if self.validator is not None:
            row = dset[0]
            self.validator(**row)

        # collate_fn
        xs = [dset[i] for i in range(min(3, len(dset)))]
        collate_fn = self.get_collate_fn(split="train")
        try:
            batch = collate_fn(xs)
            if self.batch_validator is not None:
                self.batch_validator(**batch)
        except Exception as e:
            raise CollateRuntimeError(
                f"Collate function failed on {type(self).__name__} dataset at runtime (validation step). "
                f"Please check the collate function for this dataset."
            ) from e

    @staticmethod
    def get_collate_fn(split: Optional[str] = None):
        return torch.utils.data.dataloader.default_collate

    def _take_subset(self, dset: HfDatasetDict) -> HfDatasetDict:
        subset_size = copy(self.subset_size)
        if isinstance(subset_size, int):
            subset_size = {split: subset_size for split in dset.keys()}
        elif isinstance(subset_size, dict):
            assert set(subset_size.keys()) == set(dset.keys())
        else:
            raise TypeError(f"subset_size must be an int or a dict, not {type(self.subset_size)}")
        rgn = np.random.RandomState(42)
        new_mapped_qa = {}
        for split, size in subset_size.items():
            ids = rgn.choice(list(range(len(dset[split]))), size=size, replace=False)
            new_mapped_qa[split] = dset[split].select(ids)

        return HfDatasetDict(new_mapped_qa)
