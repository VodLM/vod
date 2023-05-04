from __future__ import annotations

import abc
import collections
from typing import Iterable

import datasets
import omegaconf
from torch.utils import data as torch_data

from raffle_ds_research.utils.config import maybe_cast_omegaconf


class DataloaderSampler(abc.ABC):
    """Abstract class for dataloader samplers."""

    @abc.abstractmethod
    def __call__(self, dataset: datasets.Dataset) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        pass


class LookupSampler(DataloaderSampler):
    """Sampler that uses a lookup table to assign weights to samples."""

    def __init__(self, key: str, lookup: dict[str, float] | omegaconf.DictConfig, default_weight: float = 1.0):
        self.key = key
        self.lookup: dict[str, float] = maybe_cast_omegaconf(lookup)  # type: ignore
        self.default_weight = default_weight

    def __call__(self, dataset: datasets.Dataset) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        features = dataset[self.key]
        weights = [self.lookup.get(feature, self.default_weight) for feature in features]
        return torch_data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )


class InverseFrequencySampler(DataloaderSampler):
    """Sampler that assigns weights inversely proportional to the frequency of the feature."""

    def __init__(self, key: str):
        self.key = key

    def __call__(self, dataset: datasets.Dataset) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        features = dataset[self.key]
        counts = collections.Counter(features)
        inverse_frequency = [1 / counts[feature] for feature in features]
        return torch_data.WeightedRandomSampler(
            weights=inverse_frequency,
            num_samples=len(dataset),
            replacement=True,
        )


class ProductSampler(DataloaderSampler):
    """Sampler that computes the product of the weights of the given samplers."""

    def __init__(self, samplers: Iterable[DataloaderSampler]):
        self.samplers = samplers

    def __call__(self, dataset: datasets.Dataset) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        samplers = [sampler(dataset) for sampler in self.samplers]
        weights = [sampler.weights for sampler in samplers]
        weights = [sum(ws) for ws in zip(*weights)]
        return torch_data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
