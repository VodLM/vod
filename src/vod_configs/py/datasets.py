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


class Templates(StrictModel):
    """Prompt templates."""

    queries: str = pydantic.Field(
        default=r"Q: {{ query }}",
        description="Template formatting queries before encoding for retrieval.",
    )
    answer: str = pydantic.Field(
        default=r"A: {{ answer }}",
        description="Template formatting answers before encoding for retrieval.",
    )
    sections: str = pydantic.Field(
        default=r"D: {{ content }}",
        description="Template formatting documents before encoding for retrieval.",
    )
    prompt: str = pydantic.Field(
        default=r"",
        description="Template formatting the prompt for generative language models. A template may include instructions, few shot examples, and specific context and questions appropriate for a given task.",  # noqa: E501
    )
    prompt_response: str = pydantic.Field(
        default=r"",
        description="Template formatting a decired prompt responses before decoding for a generative task",
    )

    @property
    def input_variables(cls) -> set[str]:
        """Return the input variables."""
        variables = set()
        for attribute_value in cls.__dict__.values():
            matches = re.findall(r"{{\s*(.*?)\s*}}", attribute_value)
            variables.update(matches)
        return variables


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

    cache_dir: Optional[str] = pydantic.Field(default=None, description="Cache directory.")
    invalidate_cache: bool = pydantic.Field(default=False, description="Whether to delete an existing cached dataset.")
    prep_map_kwargs: dict[str, Any] = pydantic.Field(default={}, description="Kwargs for `datasets.map(...)`.")
    subset_size: Optional[int] = pydantic.Field(default=None, description="Take a subset of the dataset.")
    filter_unused_sections: bool = pydantic.Field(
        default=False, description="Filter out sections that are not used in the QA split."
    )
    min_section_tokens: Optional[int] = pydantic.Field(
        default=None, description="Filter out sections with less than `min_section_tokens` tokens."
    )
    group_hash_key: str = pydantic.Field(
        default="group_hash", description="Key used to store the group hash in the dataset."
    )
    group_keys: list[str] = pydantic.Field(
        default=["kb_id", "language"], description="Keys used to compute the group hash."
    )

    # validators
    _validate_prep_map_kwargs = pydantic.validator("prep_map_kwargs", allow_reuse=True, pre=True)(as_pyobj_validator)
    # _validate_templates = pydantic.validator("templates", allow_reuse=True, pre=True)(as_pyobj_validator)

    def __add__(self, other: None | DatasetOptionsDiff) -> DatasetOptions:
        """Add two options."""
        if other is None:
            return self
        attrs = other.model_dump(exclude_none=True)
        # if "templates" in attrs:
        #     attrs["templates"] = Templates(**attrs["templates"])
        new_self = self.model_copy(update=attrs)
        return new_self


class BaseDatasetConfig(StrictModel):
    """Defines a dataset."""

    name: str = pydantic.Field(
        default=None,
        description="Name of the dataset, or descriptor with pattern `name.subset:split`.",
    )
    path: str = pydantic.Field(
        default=None,
        description="Path to the dataset (overrides `name`)",
    )
<<<<<<< HEAD
    subsets: list[str] = pydantic.Field(
        default=[],
        description="A list of subset names to load.",
=======
    split: Literal["train", "val", "test", "all"] = pydantic.Field("all", description="Dataset split (train, etc.)")
    path: Optional[str] = pydantic.Field(
        None,
        description="Path to the dataset loader.",
    )
    parts: Optional[list[BaseDatasetConfig]] = pydantic.Field(
        None,
        description="Sub datasets to be concatenated. When set to None, the dataset is a single part (itself)",
>>>>>>> main
    )
    splits: list[str] = pydantic.Field(
        default=[],
        description="A list of splits to load.",
    )
    templates: Templates = pydantic.Field(Templates(), description="A set of templates used at preprocessing time.")
    options: DatasetOptions = pydantic.Field(
        DatasetOptions(),
        description="Loading/preprocessing options.",
    )

    _available_subsets: list[str] = pydantic.PrivateAttr(default=[])
    _available_splits: list[str] = pydantic.PrivateAttr(default=[])

    @property
    def descriptor(self) -> str:
        """Return the dataset descriptor `name.subset:split`."""
        desc = self.name
        if self.subsets is not None:
            desc += f".{self.subsets}"

        return f"{desc}:{self.splits}"

    def __hash__(self) -> int:
        """Hash the object based on its name and split."""
        return hash(self.descriptor)

    # Validators
    _validate_options = pydantic.validator("options", allow_reuse=True, pre=True)(as_pyobj_validator)

    # @pydantic.root_validator(pre=True)
    # def _validate_all(cls, values: dict) -> dict:
    #     return _parse_dataset_descriptor(values)

    @pydantic.model_validator(mode="before")
    def validate_name_or_path(cls, data: dict) -> dict:
        """Validate the dataset name or path."""
        if not data.get("name") and data.get("path"):
            data["name"] = data["path"]
        if not data.get("name") and not data.get("path"):
            raise ValueError("Either `name` or `path` should be provided")
        return data

    @pydantic.field_validator("name")
    @classmethod
    def validate_dset_exists(cls, value: str) -> str:
        """Validate the dataset exists."""
        # Assuming datasets.get_dataset_config_names and datasets.get_dataset_split_names are accessible
        try:
            datasets.get_dataset_config_names(value)
        except FileNotFoundError as e:
            raise ValueError(f"Dataset {value} not found") from e

        return value

    @pydantic.model_validator(mode="after")
    def validate_subset(self) -> "BaseDatasetConfig":
        """Validate the dataset subsets."""
        self._available_subsets = datasets.get_dataset_config_names(self.name)
        self._available_splits = datasets.get_dataset_split_names(self.name, self._available_subsets[0])

        if not self.subsets:
            self.subsets = self._available_subsets

        invalid_subsets = set(self.subsets) - set(self._available_subsets)
        if invalid_subsets:
            raise ValueError(
                f"Subsets {list(invalid_subsets)} not available for dataset. Available subsets: {self._available_subsets}"
            )

        if not self.splits:
            self.splits = self._available_splits

        invalid_splits = set(self.splits) - set(self._available_splits)
        if invalid_splits:
            raise ValueError(
                f"Splits {list(invalid_splits)} not available for dataset. Available splits: {self._available_splits}"
            )

        return self

    # @pydantic.model_validator(mode="after")
    # def validate_templates(self) -> "BaseDatasetConfig":
    #     """Validate the templates."""
    #     ds_builder = datasets.load_dataset_builder(self.name, self._available_subsets[-1])
    #     available_fields = ds_builder.info.features.keys()

    #     invalid_fields = set(self.templates.input_variables) - set(available_fields)
    #     if invalid_fields:
    #         raise ValueError(f"Invalid fields {invalid_fields} in Templates. Available fields: {available_fields}")

    #     return self

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
