from __future__ import annotations

import abc
import collections
from typing import Iterable

import omegaconf
from torch.utils import data as torch_data
from vod_tools.misc.config import maybe_cast_omegaconf

from src import vod_configs
from src.vod_tools import dstruct


class SamplerFactory(abc.ABC):
    """Abstract class for dataloader samplers."""

    @abc.abstractmethod
    def __call__(self, dataset: dstruct.SizedDataset[dict]) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        pass


class LookupSamplerFactory(SamplerFactory):
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


class InverseFrequencySamplerFactory(SamplerFactory):
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


class ProductSamplerFactory(SamplerFactory):
    """Sampler that computes the product of the weights of the given samplers."""

    def __init__(self, samplers: Iterable[SamplerFactory]):
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


def sampler_factory(
    config: dict
    | omegaconf.DictConfig
    | vod_configs.dataloaders.SamplerFactoryConfig
    | list[dict | omegaconf.DictConfig]
    | list[vod_configs.dataloaders.SamplerFactoryConfig],
) -> SamplerFactory:
    """Return a dataloader sampler from the given config."""
    if isinstance(config, (omegaconf.DictConfig, omegaconf.ListConfig)):
        config = omegaconf.OmegaConf.to_container(config, resolve=True)  # type: ignore

    if isinstance(config, list):
        return ProductSamplerFactory([sampler_factory(sub_config) for sub_config in config])

    if not isinstance(config, vod_configs.dataloaders.SamplerFactoryConfig):
        config = vod_configs.dataloaders.SamplerFactoryConfig(**config)

    if config.mode == "lookup":
        if config.lookup is None:
            raise ValueError("Lookup sampler requires a lookup table.")
        return LookupSamplerFactory(key=config.key, lookup=config.lookup, default_weight=config.default_weight)

    if config.mode == "inverse_frequency":
        return InverseFrequencySamplerFactory(key=config.key)

    raise ValueError(f"Unknown sampler mode: {config.mode}")
