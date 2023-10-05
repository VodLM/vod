import typing as typ

import omegaconf as omg
import pydantic
import vod_configs as vcfg
from typing_extensions import Self, Type

from .datasets import ExperimentDatasets
from .parsing import parse_experiment_datasets


class DataLoaderConfigs(vcfg.StrictModel):
    """Configures the `torch.utils.data.Dataloader` for the train, eval, and predict stages."""

    train: vcfg.DataLoaderConfig
    eval: vcfg.DataLoaderConfig
    benchmark: vcfg.DataLoaderConfig
    predict: vcfg.DataLoaderConfig

    @pydantic.field_validator("train", "eval", "benchmark", "predict", mode="before")
    @classmethod
    def _validate_dataloaders(cls: Type[Self], v: dict[str, typ.Any]) -> dict[str, typ.Any]:
        if isinstance(v, omg.DictConfig):
            return omg.OmegaConf.to_container(v)  # type: ignore
        return v


class CollateConfigs(vcfg.StrictModel):
    """Configures the collate functions for the train, eval, and static data loaders."""

    train: vcfg.RetrievalCollateConfig
    benchmark: vcfg.RetrievalCollateConfig
    predict: vcfg.BaseCollateConfig

    @pydantic.field_validator("train", "benchmark", "predict", mode="before")
    @classmethod
    def _validate_collates(cls: Type[Self], v: dict[str, typ.Any]) -> dict[str, typ.Any]:
        if isinstance(v, omg.DictConfig):
            return omg.OmegaConf.to_container(v)  # type: ignore
        return v


class Experiment(vcfg.StrictModel):
    """Configures an experiment."""

    datasets: ExperimentDatasets
    tokenizer: vcfg.TokenizerConfig
    dataloaders: DataLoaderConfigs
    collates: CollateConfigs
    trainer: vcfg.TrainerConfig
    benchmark: vcfg.BenchmarkConfig
    batch_size: vcfg.BatchSizeConfig
    sys: vcfg.SysConfig

    @classmethod
    def parse(cls: Type[Self], config: typ.Mapping[str, typ.Any]) -> Self:  # type: ignore
        """Parse an omegaconf config into a `RunConfig` instance."""
        if isinstance(config, omg.DictConfig):
            # NOTE: Resolve omegaconf variables at the top level
            config: typ.Mapping[str, typ.Any] = omg.OmegaConf.to_container(config, resolve=True)  # type: ignore
        exp_datasets = parse_experiment_datasets(config["datasets"])
        return cls(
            datasets=exp_datasets,
            trainer=vcfg.TrainerConfig(**config["trainer"]),
            benchmark=vcfg.BenchmarkConfig(**config["benchmark"]),
            batch_size=vcfg.BatchSizeConfig(**config["batch_size"]),
            dataloaders=DataLoaderConfigs(**config["dataloaders"]),
            collates=CollateConfigs(**config["collates"]),
            sys=vcfg.SysConfig(**config["sys"]),
            tokenizer=vcfg.TokenizerConfig(**config["tokenizer"]),
        )
