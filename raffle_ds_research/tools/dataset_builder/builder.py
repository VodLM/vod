from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Protocol, Type, TypeVar, Union

import datasets
import pydantic
import torch
from pydantic import ValidationError


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
        return cls._collate_config


class ExampleValidationError(ValidationError):
    ...


class CollateRuntimeError(RuntimeError):
    ...
