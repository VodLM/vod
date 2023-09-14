import abc
import collections
import typing as typ

import omegaconf
import vod_configs
import vod_types as vt
from torch.utils import data as torch_data
from vod_tools.misc.config import maybe_cast_omegaconf


class DlSamplerFactory(abc.ABC):
    """Abstract class for dataloader samplers."""

    @abc.abstractmethod
    def __call__(self, dataset: vt.DictsSequence) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        pass


class LookupDlSamplerFactory(DlSamplerFactory):
    """Sampler that uses a lookup table to assign weights to samples."""

    def __init__(self, key: str, lookup: dict[str, float] | omegaconf.DictConfig, default_weight: float = 1.0):
        self.key = key
        self.lookup: dict[str, float] = maybe_cast_omegaconf(lookup)  # type: ignore
        self.default_weight = default_weight

    def __call__(self, dataset: vt.DictsSequence) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        features = (row[self.key] for row in dataset)
        weights = [self.lookup.get(feature, self.default_weight) for feature in features]
        return torch_data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )


class InverseFrequencyDlSamplerFactory(DlSamplerFactory):
    """Sampler that assigns weights inversely proportional to the frequency of the feature."""

    def __init__(self, key: str):
        self.key = key

    def __call__(self, dataset: vt.DictsSequence) -> torch_data.WeightedRandomSampler:
        """Return a `torch.utils.data.DataLoader` for the given dataset."""
        features = (row[self.key] for row in dataset)
        counts = collections.Counter(features)
        inverse_frequency = [1 / counts[feature] for feature in features]
        return torch_data.WeightedRandomSampler(
            weights=inverse_frequency,
            num_samples=len(dataset),
            replacement=True,
        )


class ProductDlSamplerFactory(DlSamplerFactory):
    """Sampler that computes the product of the weights of the given samplers."""

    def __init__(self, samplers: typ.Iterable[DlSamplerFactory]):
        self.samplers = samplers

    def __call__(self, dataset: vt.DictsSequence) -> torch_data.WeightedRandomSampler:
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
    config: typ.Mapping[str, typ.Any]
    | omegaconf.DictConfig
    | vod_configs.SamplerFactoryConfig
    | list[dict | omegaconf.DictConfig]
    | list[vod_configs.SamplerFactoryConfig],
) -> DlSamplerFactory:
    """Return a dataloader sampler from the given config."""
    if isinstance(config, (omegaconf.DictConfig, omegaconf.ListConfig)):
        config = omegaconf.OmegaConf.to_container(config, resolve=True)  # type: ignore

    if isinstance(config, list):
        return ProductDlSamplerFactory([sampler_factory(sub_config) for sub_config in config])

    if not isinstance(config, vod_configs.SamplerFactoryConfig):
        config = vod_configs.SamplerFactoryConfig(**config)

    if config.mode == "lookup":
        if config.lookup is None:
            raise ValueError("Lookup sampler requires a lookup table.")
        return LookupDlSamplerFactory(key=config.key, lookup=config.lookup, default_weight=config.default_weight)

    if config.mode == "inverse_frequency":
        return InverseFrequencyDlSamplerFactory(key=config.key)

    raise ValueError(f"Unknown sampler mode: {config.mode}")
