from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Protocol, Type, TypeVar, Union

import datasets
import pydantic
import torch


class DatasetProtocol(Protocol):
    """Defines a dataset."""

    def __getitem__(self, index: int) -> dict:
        """Fetch a row."""
        ...

    def __len__(self) -> int:
        """Return the number of rows."""
        ...


class CollateFnProtocol(Protocol):
    """Defines a collate function."""

    def __call__(self, batch: List[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Convert a list of examples into a batch."""
        ...


D = TypeVar("D", bound=DatasetProtocol)
DD = TypeVar("DD", bound=Union[dict[str, DatasetProtocol], datasets.DatasetDict])
DorDD = TypeVar("DorDD", bound=Union[DatasetProtocol, dict[str, DatasetProtocol], datasets.DatasetDict])
CollateCfg = TypeVar("CollateCfg", bound=pydantic.BaseModel)


class DatasetBuilder(Generic[DD, CollateCfg], ABC):
    """Abstract object for all dataset datasets."""

    _collate_config: Type[CollateCfg]

    @abstractmethod
    def __call__(self) -> DD:
        """Instantiate dataset and return it."""
        ...

    def get_corpus(self) -> Optional[datasets.Dataset]:
        """Return corpus dataset if it exists."""
        return None

    @abstractmethod
    def get_collate_fn(self, config: Optional[CollateCfg] = None) -> CollateFnProtocol:
        """Return collate function for that dataset."""
        return torch.utils.data.dataloader.default_collate

    @property
    def collate_config(cls) -> Type[CollateCfg]:
        """Return the config class for the collate function."""
        return cls._collate_config
