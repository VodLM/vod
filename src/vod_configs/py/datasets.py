from __future__ import annotations

import collections
import re
from typing import Any, Iterable, Literal, Optional, TypeVar

import datasets
import omegaconf
import pydantic
import vod_datasets
from typing_extensions import Self, Type
from vod_tools.misc.config import as_pyobj_validator

from .search import MutliSearchFactoryConfig, MutliSearchFactoryDiff, SearchFactoryDefaults
from .utils import StrictModel

DsetDescriptorRegex = re.compile(r"^(?P<name>[A-Za-z_]+)(|.(?P<subset>[A-Za-z_]+))(|:(?P<split>[A-Za-z_]+))$")


class AllowMutations:
    """Context manager to allow mutations on a pydantic model."""

    def __init__(self, model: pydantic.BaseModel) -> None:
        self.model = model
        self._mutate = model.Config.allow_mutation

    def __enter__(self) -> None:
        self.model.Config.allow_mutation = True

    def __exit__(self, *args: Any) -> None:
        self.model.Config.allow_mutation = self._mutate


class Templates(StrictModel):
    """Prompt templates."""

    queries: str = r"Q: {{ query }}"
    sections: str = r"{% if title %}Title: {{ title }}. {% endif %}D: {{ section }}"


class DatasetOptionsDiff(StrictModel):
    """Preprocessing options diff."""

    templates: Optional[Templates] = None
    cache_dir: Optional[str] = None
    invalidate_cache: Optional[bool] = None
    prep_map_kwargs: Optional[dict[str, Any]] = None
    subset_size: Optional[int] = None
    filter_unused_sections: Optional[bool] = None
    min_section_tokens: Optional[int] = None
    group_hash_key: Optional[str] = None
    group_keys: Optional[list[str]] = None

    # validators
    _validate_prep_map_kwargs = pydantic.validator("prep_map_kwargs", allow_reuse=True, pre=True)(as_pyobj_validator)
    _validate_templates = pydantic.validator("templates", allow_reuse=True, pre=True)(as_pyobj_validator)


class DatasetOptions(StrictModel):
    """Preprocessing options."""

    templates: Templates = pydantic.Field(Templates(), description="A prompt template used at preprocessing time.")
    cache_dir: Optional[str] = pydantic.Field(None, description="Cache directory.")
    invalidate_cache: bool = pydantic.Field(False, description="Whether to delete an existing cached dataset.")
    prep_map_kwargs: dict[str, Any] = pydantic.Field({}, description="Kwargs for `datasets.map(...)`.")
    subset_size: Optional[int] = pydantic.Field(None, description="Take a subset of the dataset.")
    filter_unused_sections: bool = pydantic.Field(
        False, description="Filter out sections that are not used in the QA split."
    )
    min_section_tokens: Optional[int] = pydantic.Field(
        None, description="Filter out sections with less than `min_section_tokens` tokens."
    )
    group_hash_key: str = pydantic.Field("group_hash", description="Key used to store the group hash in the dataset.")
    group_keys: list[str] = pydantic.Field(["kb_id", "language"], description="Keys used to compute the group hash.")

    # validators
    _validate_prep_map_kwargs = pydantic.validator("prep_map_kwargs", allow_reuse=True, pre=True)(as_pyobj_validator)
    _validate_templates = pydantic.validator("templates", allow_reuse=True, pre=True)(as_pyobj_validator)

    def __add__(self, other: None | DatasetOptionsDiff) -> DatasetOptions:
        """Add two options."""
        if other is None:
            return self
        attrs = other.dict(exclude_none=True)
        if "templates" in attrs:
            attrs["templates"] = Templates(**attrs["templates"])
        new_self = self.copy(update=attrs)
        return new_self


class BaseDatasetConfig(StrictModel):
    """Defines a dataset."""

    name: str = pydantic.Field(
        ...,
        description="Name of the dataset, or descriptor with pattern `name.subset:split`.",
    )
    subset: Optional[str] = pydantic.Field(
        None,
        description="Dataset subset name",
    )
    path: Optional[str] = pydantic.Field(
        None,
        description="Path to the dataset loader.",
    )
    split: Literal["train", "val", "test", "all"] = pydantic.Field("all", description="Dataset split (train, etc.)")
    parts: Optional[list[BaseDatasetConfig]] = pydantic.Field(
        None,
        description="Sub datasets to be concatenated. When set to None, the dataset is a single part (itself)",
    )
    options: DatasetOptions = pydantic.Field(
        DatasetOptions(),
        description="Loading/preprocessing options.",
    )
    field: Literal["query", "section"] = pydantic.Field(
        None,
        description="Field name (used as attribute prefix - e.g., `section.input_ids`).",
    )

    @property
    def descriptor(self) -> str:
        """Return the dataset descriptor `name.subset:split`."""
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
    def parse(
        cls: Type[Self],
        config_or_descriptor: str | dict,
        *,
        base_options: Optional[DatasetOptions] = None,
        base_search: Optional[MutliSearchFactoryConfig] = None,
    ) -> Self:
        """Parse a config dictionary or dataset name into a structured config."""
        parsed = _parse_dataset_descriptor(config_or_descriptor)
        parts = parsed.pop("parts", None)
        if parts:
            if not isinstance(parts, (list, omegaconf.ListConfig)):
                raise TypeError(f"Expected `list`, found `{type(parts)}`")
            parsed["parts"] = [cls.parse(part, base_options=base_options) for part in parts]
            counts = collections.Counter(parsed["parts"])
            if max(counts.values()) > 1:
                raise ValueError(f"Found duplicated parts: {counts}")

        # parse `options` if provided
        base_options = base_options or DatasetOptions()
        options = DatasetOptionsDiff(**(parsed["options"] if "options" in parsed else {}))
        parsed["options"] = base_options + options

        # parse `search` if provided, and combine with the base search config
        if base_search is not None:
            if "search" in parsed:
                base_search = base_search + MutliSearchFactoryDiff.parse(parsed["search"])
            parsed["search"] = base_search

        return cls(**parsed)


class QueriesDatasetConfig(BaseDatasetConfig):
    """Defines a query dataset."""

    field: Literal["query"] = "query"
    link: Optional[str] = pydantic.Field(None, description="Dataset to search into (descriptor)")

    @pydantic.validator("link", pre=True)
    def _validate_link(cls, value: Optional[str]) -> Optional[str]:
        if isinstance(value, str) and ":" not in value:
            return f"{value}:all"

        return None


class SectionsDatasetConfig(BaseDatasetConfig):
    """Defines a section dataset."""

    field: Literal["section"] = "section"
    search: Optional[MutliSearchFactoryConfig] = pydantic.Field(
        None,
        description="Search config diffs for this dataset",
    )


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


BDC = TypeVar("BDC", bound=BaseDatasetConfig)


def _parse_list_dset_configs(
    x: dict | omegaconf.DictConfig | list[dict] | omegaconf.ListConfig,
    cls: Type[BDC],
    **kwargs: Any,
) -> list[BDC]:
    if isinstance(x, (omegaconf.DictConfig, omegaconf.ListConfig)):
        x = omegaconf.OmegaConf.to_container(x, resolve=True)  # type: ignore

    if isinstance(x, dict):
        return [cls.parse(x, **kwargs)]

    if isinstance(x, list):
        return [cls.parse(y, **kwargs) for y in x]

    raise ValueError(f"Unknown type `{type(x)}`")


class TrainValQueriesConfig(StrictModel):
    """Models the training and validation queries."""

    train: list[QueriesDatasetConfig]
    val: list[QueriesDatasetConfig]

    @classmethod
    def parse(
        cls: Type[Self],
        config: dict | omegaconf.DictConfig,
        base_options: Optional[DatasetOptions] = None,
    ) -> Self:
        """Parse dict or omegaconf.DictConfig into a structured config."""
        base_options = base_options or DatasetOptions()
        if "options" in config:
            base_options = base_options + DatasetOptionsDiff(**config["options"])

        return cls(
            train=_parse_list_dset_configs(config["train"], cls=QueriesDatasetConfig, base_options=base_options),
            val=_parse_list_dset_configs(config["val"], cls=QueriesDatasetConfig, base_options=base_options),
        )


class SectionsConfig(StrictModel):
    """Models the sections."""

    sections: list[SectionsDatasetConfig]

    @classmethod
    def parse(
        cls: Type[Self],
        config: dict | omegaconf.DictConfig,
        base_options: Optional[DatasetOptions] = None,
        base_search: Optional[MutliSearchFactoryConfig] = None,
    ) -> Self:
        """Parse dict or omegaconf.DictConfig into a structured config."""
        base_options = base_options or DatasetOptions()
        if "options" in config:
            base_options = base_options + DatasetOptionsDiff(**config["options"])

        return cls(
            sections=_parse_list_dset_configs(
                config["sections"],
                cls=SectionsDatasetConfig,
                base_options=base_options,
                base_search=base_search,
            )
        )


class TrainDatasetsConfig(StrictModel):
    """Defines the training datasets."""

    queries: TrainValQueriesConfig
    sections: SectionsConfig

    @classmethod
    def parse(
        cls: Type[Self],
        config: dict | omegaconf.DictConfig,
        base_options: Optional[DatasetOptions] = None,
        base_search: Optional[MutliSearchFactoryConfig] = None,
    ) -> Self:
        """Parse dict or omegaconf.DictConfig into a structured config."""
        base_options = base_options or DatasetOptions()
        if "options" in config:
            base_options = base_options + DatasetOptionsDiff(**config["options"])

        sections = SectionsConfig.parse(
            config["sections"],
            base_options=base_options,
            base_search=base_search,
        )

        queries = TrainValQueriesConfig.parse(
            config["queries"],
            base_options=base_options,
        )

        # Implicitely link the queries to the sections when there is only one section dataset
        if len(sections.sections) == 1:
            for query in queries.train + queries.val:
                with AllowMutations(query):
                    query.link = sections.sections[0].descriptor

        return cls(
            queries=queries,
            sections=sections,
        )


class BenchmarkDatasetConfig(StrictModel):
    """Defines a benchmark."""

    queries: QueriesDatasetConfig
    sections: SectionsDatasetConfig

    @classmethod
    def parse(
        cls: Type[Self],
        config: dict | omegaconf.DictConfig,
        base_options: Optional[DatasetOptions] = None,
        base_search: Optional[MutliSearchFactoryConfig] = None,
    ) -> Self:
        """Parse dict or omegaconf.DictConfig into a structured config."""
        base_options = base_options or DatasetOptions()
        if "options" in config:
            base_options = base_options + DatasetOptionsDiff(**config["options"])

        sections = SectionsDatasetConfig.parse(
            config["sections"],
            base_options=base_options,
            base_search=base_search,
        )

        queries = QueriesDatasetConfig.parse(
            config["queries"],
            base_options=base_options,
        )

        # Implicitely link the queries to the sections when there is only one section dataset
        with AllowMutations(queries):
            queries.Config.allow_mutation = True
            queries.link = sections.descriptor

        return cls(
            queries=queries,
            sections=sections,
        )


class DatasetsConfig(StrictModel):
    """Deine all datasets, including the base search config."""

    training: TrainDatasetsConfig
    benchmark: list[BenchmarkDatasetConfig]

    @classmethod
    def parse(
        cls: Type[Self],
        config: dict | omegaconf.DictConfig,
        base_options: Optional[DatasetOptions] = None,
    ) -> Self:
        """Parse dict or omegaconf.DictConfig into a structured config."""
        base_options = base_options or DatasetOptions()
        if "options" in config:
            base_options = base_options + DatasetOptionsDiff(**config["options"])

        # parse the search defaults
        search_defaults = SearchFactoryDefaults.parse(config["search_defaults"])

        # parse the base search config
        base_search = search_defaults + MutliSearchFactoryDiff.parse(config["search"])

        return cls(
            training=TrainDatasetsConfig.parse(
                config["training"],
                base_options=base_options,
                base_search=base_search,
            ),
            benchmark=[
                BenchmarkDatasetConfig.parse(
                    cfg,
                    base_options=base_options,
                    base_search=base_search,
                )
                for cfg in config["benchmark"]
            ],
        )

    def get_dataset_configs(  # noqa: C901
        self,
        what: None | Literal["all", "queries", "sections"] = None,
        split: None | Literal["all", "train", "val", "train+val", "benchmark"] = None,
    ) -> Iterable[BaseDatasetConfig]:
        """Iterate over the dataset configs."""
        what = what or "all"
        split = split or "all"
        if split in ["train", "train+val", "all"]:
            if what in ["all", "queries"]:
                yield from self.training.queries.train
            if what in ["all", "sections"]:
                yield from self.training.sections.sections

        if split in ["val", "train+val", "all"]:
            if what in ["all", "queries"]:
                yield from self.training.queries.val
            if what in ["all", "sections"]:
                yield from self.training.sections.sections

        if split in ["benchmark", "all"]:
            for benchmark in self.benchmark:
                if what in ["all", "queries"]:
                    yield benchmark.queries
                if what in ["all", "sections"]:
                    yield benchmark.sections

    def load_datasets(
        self,
        what: None | Literal["all", "queries", "sections"] = None,
        split: None | Literal["all", "train", "val", "train+val", "benchmark"] = None,
    ) -> Iterable[datasets.Dataset]:
        """Load the datasets."""
        for config in self.get_dataset_configs(what=what, split=split):
            yield vod_datasets.load_dataset(config)
