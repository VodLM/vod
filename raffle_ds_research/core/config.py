from __future__ import annotations

import copy
import dataclasses
import functools
import pathlib
from typing import Any, Literal, Optional, Union

import hydra
import omegaconf
import pydantic
import transformers
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Self, Type

from raffle_ds_research.core.mechanics.dataloader_sampler import DataloaderSampler, dl_sampler_factory
from raffle_ds_research.core.mechanics.search_engine import SearchConfig
from raffle_ds_research.core.workflows.utils.schedule import BaseSchedule, schedule_factory
from raffle_ds_research.tools.utils.loader_config import DataLoaderConfig
from raffle_ds_research.utils.config import as_pyobj_validator

_DEFAULT_SPLITS = ["train", "validation"]

QUESTION_TEMPLATE = "Question: {{ text }}"
SECTION_TEMPLATE = "{% if title %}Title: {{ title }}. Document: {% endif %}{{ content }}"
DEFAULT_TEMPLATES = {
    "question": QUESTION_TEMPLATE,
    "section": SECTION_TEMPLATE,
}

_DEFAULT_SPLITS = ["train", "validation"]
_N_VALID_SAMPLES = 3


class NamedDset(pydantic.BaseModel):
    """A dataset name with splits."""

    class Config:
        """Pydantic configuration."""

        extra = pydantic.Extra.forbid

    name: str
    split: str

    @property
    def split_alias(self) -> str:
        """Return a slightluy more human-readable version of the split name."""
        aliases = {
            "validation": "val",
        }
        return aliases.get(self.split, self.split)

    @pydantic.validator("split", pre=True)
    def _validate_split(cls, v: str) -> str:
        dictionary = {
            "train": "train",
            "val": "validation",
            "validation": "validation",
            "test": "test",
        }
        if v not in dictionary:
            raise ValueError(f"Invalid split name: {v}")
        return dictionary[v]

    def __hash__(self) -> int:
        """Hash the object based on its name and split."""
        return hash((self.name, self.split))


def parse_named_dsets(names: str | list[str], default_splits: Optional[list[str]] = None) -> list[NamedDset]:
    """Parse a string of dataset names.

    Names are `+` separated and splits are specified with `:` and separated by `-`.
    """
    if default_splits is None:
        default_splits = copy.copy(_DEFAULT_SPLITS)

    if not isinstance(names, (list, omegaconf.ListConfig)):
        names = [names]

    outputs = []
    for part in (p for parts in names for p in parts.split("+")):
        if ":" in part:
            name, splits = part.split(":")
            splits = splits.split("-")
        else:
            name = part
            splits = default_splits
        for split in splits:
            outputs.append(NamedDset(name=name, split=split))
    return outputs


class BaseDatasetFactoryConfig(pydantic.BaseModel):
    """Defines a base configuration for a retrieval dataset builder."""

    class Config:
        """Pydantic config for the `DatasetFactoryConfig` class."""

        extra = pydantic.Extra.forbid
        arbitrary_types_allowed = True

    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]
    templates: dict[str, str] = DEFAULT_TEMPLATES
    prep_map_kwargs: dict[str, Any] = {}
    subset_size: Optional[int] = None
    filter_unused_sections: bool = True
    group_hash_key: str = "group_hash"
    group_keys: list[str] = ["kb_id", "language"]

    # validators
    _validate_templates = pydantic.validator("templates", allow_reuse=True, pre=True)(as_pyobj_validator)
    _validate_prep_map_kwargs = pydantic.validator("prep_map_kwargs", allow_reuse=True, pre=True)(as_pyobj_validator)


class DatasetFactoryConfig(NamedDset, BaseDatasetFactoryConfig):
    """Defines a configuration for a retrieval dataset builder."""

    ...


class MultiDatasetFactoryConfig(BaseDatasetFactoryConfig):
    """Defines a configuration for a retrieval dataset builder."""

    # dataset groups
    train: list[NamedDset]
    validation: list[NamedDset]
    benchmark: list[NamedDset]

    # benchmark metrics
    metrics: list[str] = ["ndcg", "mrr", "hitrate@01", "hitrate@03", "hitrate@10", "hitrate@30"]

    # validators
    _validate_train = pydantic.validator("train", allow_reuse=True, pre=True)(
        functools.partial(parse_named_dsets, default_splits=["train"])
    )
    _validate_validation = pydantic.validator("validation", allow_reuse=True, pre=True)(
        functools.partial(parse_named_dsets, default_splits=["validation"])
    )
    _validate_benchmark = pydantic.validator("benchmark", allow_reuse=True, pre=True)(
        functools.partial(parse_named_dsets, default_splits=["test"])
    )
    _validate_metrics = pydantic.validator("metrics", allow_reuse=True, pre=True)(as_pyobj_validator)

    @classmethod
    def parse(cls: Type[Self], obj: dict | omegaconf.DictConfig | Self) -> Self:
        """Parse a benchmark config."""
        if isinstance(obj, cls):
            return obj

        return hydra.utils.instantiate(obj)

    def dataset_factory_config(self, dset: NamedDset) -> DatasetFactoryConfig:
        """Returns the `DatasetFactoryConfig` for a given dataset."""
        return DatasetFactoryConfig(
            name=dset.name,
            split=dset.split,
            **self.dict(exclude={"train", "validation", "benchmark", "metrics"}),
        )

    def get(self, what: Literal["all", "train", "validation", "benchmark"]) -> set[NamedDset]:
        """Return all datasets."""
        known_groups = {
            "all": self.train + self.validation + self.benchmark,
            "train": self.train,
            "validation": self.validation,
            "benchmark": self.benchmark,
        }
        try:
            groups = known_groups[what]
        except KeyError as exc:
            raise ValueError(f"Invalid group: {what}. Valid groups are: {list(known_groups.keys())}") from exc

        return set(groups)


class KeyMap(pydantic.BaseModel):
    """Defines the name of the keys used on the query side and on the section side."""

    class Config:
        """Pydantic config."""

        extra = "forbid"

    query: str
    section: str


class BaseCollateConfig(pydantic.BaseModel):
    """Defines a base configuration for the collate function."""

    class Config:
        """Pydantic config."""

        extra = "forbid"

    question_max_length: int = 512
    section_max_length: int = 512


class RetrievalCollateConfig(BaseCollateConfig):
    """Defines a configuration for the retrieval collate function."""

    class Config:
        """Pydantic config."""

        extra = "forbid"

    # base config
    n_sections: int = 10
    prefetch_n_sections: int = 100
    max_pos_sections: int = 3
    post_filter: Optional[str] = None
    do_sample: bool = False
    in_batch_negatives: bool = False
    in_batch_neg_offset: int = 0
    prep_num_proc: int = 4

    # name of the keys to use on the query side and on the section side
    text_keys: KeyMap = KeyMap(query="text", section="text")  #  text field
    vector_keys: KeyMap = KeyMap(query="vector", section="vector")  #  vector field
    section_id_keys: KeyMap = KeyMap(query="section_ids", section="id")  #  label field (section ids)
    group_id_keys: KeyMap = KeyMap(query="group_hash", section="group_hash")  #  group hash (kb_id, lang, etc.)


class ScheduleConfig(pydantic.BaseModel):
    """Configures a group of indexes (e.g., bm25, faiss)."""

    period: Union[int, list[int]]
    reset_model_on_period_start: bool = False
    benchmark_on_init: bool = True
    parameters: dict[str, BaseSchedule] = {}

    # validators
    _validate_update_freq = pydantic.validator("period", allow_reuse=True, pre=True)(as_pyobj_validator)

    @pydantic.validator("parameters", pre=True)
    def _validate_parameters(cls, x: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if x is None:
            return {}

        params = {}
        for k, v in x.items():
            if isinstance(v, (dict, omegaconf.DictConfig)):
                params[k] = schedule_factory(**v)
            elif isinstance(v, (float, int)):
                params[k] = schedule_factory(mode="constant", value=v)
            else:
                params[k] = v

        return params

    @classmethod
    def parse(cls: Type[Self], obj: dict | Self) -> Self:
        """Parse a benchmark config."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, omegaconf.DictConfig):
            return hydra.utils.instantiate(obj)

        return cls(**obj)  # type: ignore

    def get_parameters(self, step: float) -> dict[str, float]:
        """Return the parameters for a given step."""
        return {k: v(step) for k, v in self.parameters.items()}


@dataclasses.dataclass
class DataLoaderConfigs:
    """Configures the `torch.utils.data.Dataloader` for the train, eval, and predict stages."""

    train: DataLoaderConfig
    eval: DataLoaderConfig
    predict: DataLoaderConfig

    @classmethod
    def parse(cls: Type[Self], config: DictConfig) -> Self:
        """Parse an omegaconf config into a DataLoaderConfigs instance."""
        return cls(
            train=DataLoaderConfig(**config.train),
            eval=DataLoaderConfig(**config.eval),
            predict=DataLoaderConfig(**config.predict),
        )


@dataclasses.dataclass
class CollateConfigs:
    """Configures the collate functions for the train, eval, and static data loaders."""

    train: RetrievalCollateConfig
    benchmark: RetrievalCollateConfig
    predict: BaseCollateConfig

    @classmethod
    def parse(cls: Type[Self], config: DictConfig) -> Self:
        """Parse an omegaconf config into a CollateConfigs instance."""
        return cls(
            train=RetrievalCollateConfig(**config.train),
            benchmark=RetrievalCollateConfig(**config.benchmark),
            predict=BaseCollateConfig(**config.predict),
        )


@dataclasses.dataclass
class SysConfig:
    """Configures the system directories."""

    raffle_path: pathlib.Path
    work_dir: pathlib.Path
    cache_dir: pathlib.Path

    @classmethod
    def parse(cls: Type[Self], config: DictConfig) -> Self:
        """Parse an omegaconf config into a CollateConfigs instance."""
        return cls(
            raffle_path=pathlib.Path(config.raffle_path),
            work_dir=pathlib.Path(config.work_dir),
            cache_dir=pathlib.Path(config.cache_dir),
        )


@dataclasses.dataclass
class SearchConfigs:
    """Models the search engines for the training and benchmark steps."""

    training: SearchConfig
    benchmark: SearchConfig

    @classmethod
    def parse(cls: Type[Self], config: DictConfig) -> Self:
        """Parse an omegaconf config into a CollateConfigs instance."""
        return cls(
            training=SearchConfig(**config.training),
            benchmark=SearchConfig(**config.benchmark),
        )


@dataclasses.dataclass
class TrainWithIndexUpdatesConfigs:
    """Models the configuration for a workflow that trains a model and periodically indexes the data."""

    dataset: MultiDatasetFactoryConfig
    dataloaders: DataLoaderConfigs
    collates: CollateConfigs
    schedule: ScheduleConfig
    search: SearchConfigs
    sys: SysConfig
    benchmark_search_overrides: Optional[dict] = None
    dl_sampler: Optional[DataloaderSampler] = None

    @classmethod
    def parse(cls: Type[Self], config: DictConfig) -> Self:
        """Parse an omegaconf config into a TrainWithIndexConfigs instance."""
        dataset = MultiDatasetFactoryConfig.parse(config.dataset)
        schedule_config = ScheduleConfig.parse(config.schedule)
        dataloaders = DataLoaderConfigs.parse(config.dataloaders)
        collates = CollateConfigs.parse(config.collates)
        sys_config = SysConfig.parse(config.sys)

        # parse the search configs. if benchmark_search is not specified, use the training search.
        training_search = SearchConfig.parse(config.search)
        if config.benchmark_search is None:
            benchmark_search = training_search
        else:
            merged_config = OmegaConf.merge(config.search, config.benchmark_search)
            benchmark_search = SearchConfig.parse(merged_config)  # type: ignore

        return cls(
            dataset=dataset,
            schedule=schedule_config,
            dataloaders=dataloaders,
            collates=collates,
            search=SearchConfigs(training=training_search, benchmark=benchmark_search),
            sys=sys_config,
            dl_sampler=dl_sampler_factory(config.dl_sampler) if config.dl_sampler is not None else None,
        )
