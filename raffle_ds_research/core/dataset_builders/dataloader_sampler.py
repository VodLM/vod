import abc
from typing import Iterable

import datasets
import collections

import omegaconf
from torch.utils import data as torch_data


class DataloaderSampler(abc.ABC):
    """Abstract class for dataloader samplers."""

    @abc.abstractmethod
    def __call__(self, dataset: datasets.Dataset) -> torch_data.WeightedRandomSampler:
        pass


class LookupSampler(DataloaderSampler):
    def __init__(self, key: str, lookup: dict[str, float], default_weight: float = 1.0):
        self.key = key
        if isinstance(lookup, omegaconf.DictConfig):
            lookup = omegaconf.OmegaConf.to_container(lookup)
        self.lookup = lookup
        self.default_weight = default_weight

    def __call__(self, dataset: datasets.Dataset) -> torch_data.WeightedRandomSampler:
        features = dataset[self.key]
        weights = [self.lookup.get(feature, self.default_weight) for feature in features]
        return torch_data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )


class InverseFrequencySampler(DataloaderSampler):
    def __init__(self, key: str):
        self.key = key

    def __call__(self, dataset: datasets.Dataset) -> torch_data.WeightedRandomSampler:
        features = dataset[self.key]
        counts = collections.Counter(features)
        inverse_frequency = [1 / counts[feature] for feature in features]
        return torch_data.WeightedRandomSampler(
            weights=inverse_frequency,
            num_samples=len(dataset),
            replacement=True,
        )


class ProductSampler(DataloaderSampler):
    def __init__(self, samplers: Iterable[DataloaderSampler]):
        self.samplers = samplers

    def __call__(self, dataset: datasets.Dataset) -> torch_data.WeightedRandomSampler:
        samplers = [sampler(dataset) for sampler in self.samplers]
        weights = [sampler.weights for sampler in samplers]
        weights = [sum(ws) for ws in zip(*weights)]
        return torch_data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )