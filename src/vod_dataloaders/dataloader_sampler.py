from __future__ import annotations

import abc
import collections
import copy
from typing import Iterable

import omegaconf
from torch.utils import data as torch_data
from vod_tools import dstruct
from vod_tools.misc.config import maybe_cast_omegaconf


class DataloaderSampler(abc.ABC):
    """Abstract class for dataloader samplers."""

    @abc.abstractmethod
    def __call__(self, dataset: dstruct.SizedDataset[dict]) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        pass


class LookupSampler(DataloaderSampler):
    """Sampler that uses a lookup table to assign weights to samples."""

    def __init__(self, key: str, lookup: dict[str, float] | omegaconf.DictConfig, default_weight: float = 1.0):
        self.key = key
        self.lookup: dict[str, float] = maybe_cast_omegaconf(lookup)  # type: ignore
        self.default_weight = default_weight

    def __call__(self, dataset: dstruct.SizedDataset[dict]) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        features = (row[self.key] for row in dataset)
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

    def __call__(self, dataset: dstruct.SizedDataset[dict]) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        features = (row[self.key] for row in dataset)
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

    def __call__(self, dataset: dstruct.SizedDataset[dict]) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        samplers = [sampler(dataset) for sampler in self.samplers]
        weights = [sampler.weights for sampler in samplers]
        weights = [sum(ws) for ws in zip(*weights)]
        return torch_data.WeightedRandomSampler(
            weights=weights,  # type: ignore
            num_samples=len(dataset),
            replacement=True,
        )


def dl_sampler_factory(config: dict | omegaconf.DictConfig) -> None | DataloaderSampler:
    """Return a dataloader sampler from the given config."""
    if isinstance(config, omegaconf.DictConfig):
        config = omegaconf.OmegaConf.to_container(config, resolve=True)  # type: ignore
    else:
        config = copy.deepcopy(config)
    if len(config) == 0:
        return None
    mode = config.pop("mode")
    if mode == "product":
        return ProductSampler([dl_sampler_factory(sub_config) for sub_config in config["samplers"]])  # type: ignore

    sampler = {
        "lookup": LookupSampler,
        "inverse_frequency": InverseFrequencySampler,
    }

    return sampler[mode](**config)
