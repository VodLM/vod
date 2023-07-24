from __future__ import annotations

import collections
import re
from typing import Any, Literal, Optional

import omegaconf
import pydantic
from typing_extensions import Self, Type
from vod_configs.py.search import MutliSearchFactoryConfig, MutliSearchFactoryDiff, SearchFactoryDefaults
from vod_configs.py.utils import StrictModel
from vod_tools.misc.config import as_pyobj_validator

QUESTION_TEMPLATE = "Question: {{ text }}"
SECTION_TEMPLATE = "{% if title %}Title: {{ title }}. Document: {% endif %}{{ content }}"
DEFAULT_TEMPLATES = {
    "question": QUESTION_TEMPLATE,
    "section": SECTION_TEMPLATE,
}


DsetDescriptorRegex = re.compile(r"^(?P<name>[A-Za-z_]+)(|.(?P<subset>[A-Za-z_]+))(|:(?P<split>[A-Za-z_]+))$")


class DatasetFactoryConfig(StrictModel):
    """Defines a base configuration for a retrieval dataset builder."""

    templates: dict[str, str] = DEFAULT_TEMPLATES
    prep_map_kwargs: dict[str, Any] = {}
    subset_size: Optional[int] = None
    filter_unused_sections: bool = False
    min_section_tokens: Optional[int] = None
    group_hash_key: str = "group_hash"
    group_keys: list[str] = ["kb_id", "language"]

    # validators
    _validate_templates = pydantic.validator("templates", allow_reuse=True, pre=True)(as_pyobj_validator)
    _validate_prep_map_kwargs = pydantic.validator("prep_map_kwargs", allow_reuse=True, pre=True)(as_pyobj_validator)


class DatasetOptions(StrictModel):
    """Preprocessing options."""

    template: Optional[str] = pydantic.Field(
        None,
        description="A prompt template used at preprocessing time.",
    )


class DatasetConfig(pydantic.BaseModel):
    """Defines a dataset."""

    name: str = pydantic.Field(
        ...,
        description="Name of the dataset following the pattern `name.subset:split`.",
    )
    subset: Optional[str] = pydantic.Field(
        None,
        description="Dataset subset name",
    )
    split: Optional[Literal["train", "val", "test", "all"]] = pydantic.Field(
        "all", description="Dataset split (train, etc.)"
    )
    parts: Optional[list[DatasetConfig]] = pydantic.Field(
        None,
        description="Sub datasets to be concatenated. When set to None, the dataset is a single part (itself)",
    )
    options: Optional[DatasetOptions] = pydantic.Field(
        None,
        description="Loading/preprocessing options.",
    )
    link: Optional[str] = pydantic.Field(None, description="Dataset to search into (descriptor)")
    search: Optional[MutliSearchFactoryDiff] = pydantic.Field(
        None,
        description="Search config diffs for this dataset",
    )

    @property
    def descriptor(self) -> str:
        """Return the dataset descriptor (code name)."""
        desc = self.name
        if self.subset is not None:
            desc += f".{self.subset}"

        return f"{desc}:{self.split}"

    def __hash__(self) -> int:
        """Hash the object based on its name and split."""
        return hash(self.descriptor)

    # Validators
    _validate_options = pydantic.validator("options", allow_reuse=True, pre=True)(as_pyobj_validator)

    @pydantic.root_validator(pre=True)
    def _validate_all(cls, values: dict) -> dict:
        return _parse_dataset_descriptor(values)

    @classmethod
    def parse(cls: Type[Self], config_or_descriptor: str | dict) -> Self:
        """Parse a config dictionary or dataset name into a structured config."""
        parsed = _parse_dataset_descriptor(config_or_descriptor)
        parts = parsed.pop("parts", None)
        if parts:
            if not isinstance(parts, (list, omegaconf.ListConfig)):
                raise TypeError(f"Expected `list`, found `{type(parts)}`")
            parsed["parts"] = [cls.parse(part) for part in parts]
            counts = collections.Counter(parsed["parts"])
            if max(counts.values()) > 1:
                raise ValueError(f"Found duplicated parts: {counts}")
        if "search" in parsed:
            parsed["search"] = MutliSearchFactoryDiff.parse(parsed["search"])
        return cls(**parsed)


def _parse_dataset_descriptor(
    config_or_name: str | dict | omegaconf.DictConfig,  # type: ignore
) -> dict:
    if isinstance(config_or_name, (omegaconf.DictConfig)):
        config_or_name: dict = omegaconf.OmegaConf.to_container(config_or_name, resolve=True)  # type: ignore

    if isinstance(config_or_name, str):
        config_or_name = {"name": config_or_name}
    if not isinstance(config_or_name, dict):
        raise TypeError(f"config_or_name should be a dict. Found `{type(config_or_name)}`")

    try:
        name = config_or_name["name"]
    except KeyError as exc:
        raise KeyError(
            f"Key `name` should be provided when parsing `DatasetConfig`. Found keys={list(config_or_name.keys())}"
        ) from exc

    parsed = DsetDescriptorRegex.match(name)
    if parsed is None:
        raise ValueError(f"Couldn't parse name `{name}` with pattern {DsetDescriptorRegex.pattern}")

    # Filter None
    parsed_ = {k: v for k, v in parsed.groupdict().items() if v is not None}
    config_ = {k: v for k, v in config_or_name.items() if v is not None}

    # Create the final config
    final_config = {}
    for key in parsed_.keys() | config_.keys():
        parsed_value = parsed_.get(key, None)
        config_value = config_.get(key, None)
        final_config[key] = parsed_value or config_value

    return final_config


def _parse_list_dset_configs(x: dict | omegaconf.DictConfig | list[dict] | omegaconf.ListConfig):
    if isinstance(x, (omegaconf.DictConfig, omegaconf.ListConfig)):
        x = omegaconf.OmegaConf.to_container(x, resolve=True)  # type: ignore

    if isinstance(x, dict):
        return [DatasetConfig.parse(x)]

    if isinstance(x, list):
        return [DatasetConfig.parse(y) for y in x]

    raise ValueError(f"Unknown type `{type(x)}`")


class TrainDatasetsConfig(StrictModel):
    """Defines the training datasets."""

    train_queries: list[DatasetConfig]
    val_queries: list[DatasetConfig]
    sections: list[DatasetConfig]

    # validators
    _validate_train_queries = pydantic.validator("train_queries", allow_reuse=True, pre=True)(_parse_list_dset_configs)
    _validate_val_queries = pydantic.validator("val_queries", allow_reuse=True, pre=True)(_parse_list_dset_configs)
    _validate_sections = pydantic.validator("sections", allow_reuse=True, pre=True)(_parse_list_dset_configs)


class BenchmarkDatasetsConfig(StrictModel):
    """Defines a benchmark."""

    queries: DatasetConfig
    sections: DatasetConfig

    # validators
    _validate_queries = pydantic.validator("queries", allow_reuse=True, pre=True)(DatasetConfig.parse)
    _validate_sections = pydantic.validator("sections", allow_reuse=True, pre=True)(DatasetConfig.parse)


class DatasetsConfig(StrictModel):
    """Deine all datasets, including the base search config."""

    train: TrainDatasetsConfig
    benchmark: list[BenchmarkDatasetsConfig]
    factory: DatasetFactoryConfig
    base_search: MutliSearchFactoryDiff

    # Validators
    _validate_train = pydantic.validator("train", allow_reuse=True, pre=True)(as_pyobj_validator)
    _validate_factory = pydantic.validator("factory", allow_reuse=True, pre=True)(as_pyobj_validator)
    _validate_base_search = pydantic.validator("base_search", allow_reuse=True, pre=True)(as_pyobj_validator)

    @pydantic.validator("benchmark", pre=True)
    def _validate_benchmark(cls, v):
        if isinstance(v, (omegaconf.DictConfig, omegaconf.ListConfig)):
            v = omegaconf.OmegaConf.to_container(v, resolve=True)  # type: ignore

        if isinstance(v, dict):
            v = [v]
        if not isinstance(v, list):
            raise TypeError(f"Unknown type {type(v)}")
        return [BenchmarkDatasetsConfig(**y) for y in v]

    # Utilities
    def resolve_search_config(
        self,
        defaults: SearchFactoryDefaults,
        config: None | MutliSearchFactoryDiff,
    ) -> MutliSearchFactoryConfig:
        return defaults + self.base_search + config
