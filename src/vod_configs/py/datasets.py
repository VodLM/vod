from __future__ import annotations

import re
from typing import Any, Iterable, Literal, Optional, Protocol, TypeVar, Union, runtime_checkable

import datasets
import omegaconf
import pydantic
from hydra.utils import instantiate
from typing_extensions import Self, Type
from vod_tools.misc.config import as_pyobj_validator

from .search import MutliSearchFactoryConfig, MutliSearchFactoryDiff, SearchFactoryDefaults
from .sectioning import SectioningConfig
from .utils import AllowMutations, StrictModel

DsetDescriptorRegex = re.compile(
    r"^(?P<identifier>[A-Za-z0-9_]+)\((?P<name_or_path>[A-Za-z0-9_/]+)(|.(?P<subset>[A-Za-z0-9_\+]+))(|:(?P<split>[A-Za-z_\+]+))\)(|\s*->\s*(?P<link>[A-Za-z0-9_]+))\s*$"  # noqa: E501
)


@runtime_checkable
class DatasetLoader(Protocol):
    """A dataset loader."""

    def __call__(self, subset: None | str = None, split: None | str = None, **kwargs: Any) -> datasets.Dataset:
        """Load a dataset."""
        ...


class DatasetOptionsDiff(StrictModel):
    """Preprocessing options diff."""

    prep_map_kwargs: Optional[dict[str, Any]] = None
    subset_size: Optional[int] = None
    sectioning: Optional[SectioningConfig] = None

    # validators
    _validate_prep_map_kwargs = pydantic.field_validator("prep_map_kwargs", mode="before")(as_pyobj_validator)


class DatasetOptions(StrictModel):
    """Preprocessing options."""

    prep_map_kwargs: dict[str, Any] = pydantic.Field(
        default_factory=dict,
        description="Kwargs for `datasets.map(...)`.",
    )
    subset_size: Optional[int] = pydantic.Field(
        default=None,
        description="Take a subset of the dataset.",
    )
    sectioning: Optional[SectioningConfig] = pydantic.Field(
        default=None,
        description="Configures a sectioning algorithm to split input documents/sections into smaller chunks.",
    )

    # validators
    _validate_prep_map_kwargs = pydantic.field_validator("prep_map_kwargs", mode="before")(as_pyobj_validator)

    def __add__(self, other: None | DatasetOptionsDiff) -> DatasetOptions:
        """Add two options."""
        if other is None:
            return self
        attrs = other.model_dump(exclude_none=True)
        new_self = self.model_copy(update=attrs)
        return new_self


class BaseDatasetConfig(StrictModel):
    """Defines a dataset."""

    _dynamic_dataset_validation = pydantic.PrivateAttr(False)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True  # <- new attribute
        extra = "forbid"
        frozen = True
        from_attributes = True

    identifier: str = pydantic.Field(
        ...,
        description="Name of the dataset, or descriptor with pattern `name.subset:split`.",
    )
    name_or_path: Union[str, DatasetLoader] = pydantic.Field(
        ...,
        description="Path to the dataset loader (overrides `name`)",
    )
    subsets: list[str] = pydantic.Field(
        default_factory=list,
        description="A list of subset names to load.",
    )
    split: str | None = pydantic.Field(
        None,
        description="Dataset split (train, etc.)",
    )
    options: DatasetOptions = pydantic.Field(
        default_factory=DatasetOptions,  # type: ignore
        description="Loading/preprocessing options.",
    )

    @property
    def descriptor(self) -> str:
        """Return the dataset descriptor `name.subset:split`."""
        desc = self.name_or_path if isinstance(self.name_or_path, str) else type(self.name_or_path).__name__
        if self.subsets is not None:
            desc += f".{'_'.join(self.subsets)}"

        return f"{self.identifier}({desc}:{self.split})"

    def __hash__(self) -> int:
        """Hash the object based on its name and split."""
        return hash(self.descriptor)

    _validate_options = pydantic.field_validator("options", mode="before")(as_pyobj_validator)

    @pydantic.field_validator("subsets", mode="before")
    def _validate_subsets(cls, value: None | str | list[str]) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @pydantic.field_validator("name_or_path", mode="before")
    def validate_name_or_path(cls, value: str) -> str:
        """Validate the dataset name or path."""
        # Assuming datasets.get_dataset_config_names are accessible
        if not isinstance(value, str):
            return value
        try:
            datasets.get_dataset_config_names(value)
            return value
        except FileNotFoundError as e:
            raise ValueError(f"Dataset {value} not found") from e

    @pydantic.model_validator(mode="after")
    def validate_subsets_and_splits(self) -> Self:
        """Validate the dataset subsets."""
        if not self._dynamic_dataset_validation:
            # disable the validation of datasets (`subset` & `split`) using the huggingface Hub.
            return self
        if not isinstance(self.name_or_path, str):
            return self
        with AllowMutations(self):
            available_subsets = datasets.get_dataset_config_names(self.name_or_path)

            if not self.subsets:
                self.subsets = available_subsets

            invalid_subsets = set(self.subsets) - set(available_subsets)
            if invalid_subsets:
                raise ValueError(
                    f"Subsets {list(invalid_subsets)} not available for dataset. "
                    f"Available subsets: `{available_subsets}`"
                )

            # Check the splits & return
            if self.split is None:
                return self
            for subset_name in self.subsets:
                available_splits = datasets.get_dataset_split_names(self.name_or_path, subset_name)
                if self.split not in available_splits:
                    raise ValueError(
                        f"Split `{self.split}` not available for dataset `{self.name_or_path}`. "
                        f"Available splits: {available_splits}"
                    )

        return self

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
    link: Optional[str] = pydantic.Field(
        None,
        description="Identifier of the `Sections` dataset to search into.",
    )

    @pydantic.field_validator("link", mode="before")
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


DatasetConfig = Union[QueriesDatasetConfig, SectionsDatasetConfig]


def _parse_dataset_descriptor(
    config_or_descriptor: str | dict | omegaconf.DictConfig,  # type: ignore
) -> dict:
    if isinstance(config_or_descriptor, omegaconf.DictConfig):
        config_or_descriptor = omegaconf.OmegaConf.to_container(config_or_descriptor, resolve=True)  # type: ignore
    if isinstance(config_or_descriptor, dict) and "name_or_path" in config_or_descriptor:
        name_or_path = config_or_descriptor["name_or_path"]
        if isinstance(name_or_path, dict):
            name_or_path = instantiate(name_or_path)
        return {
            **config_or_descriptor,
            "name_or_path": name_or_path,
        }

    if isinstance(config_or_descriptor, str):
        config_or_descriptor = {"descriptor": config_or_descriptor}
    if not isinstance(config_or_descriptor, dict):
        raise TypeError(f"`config_or_descriptor` should be a dict. Found `{type(config_or_descriptor)}`")

    if descriptor := config_or_descriptor.get("descriptor", None):
        parsed = DsetDescriptorRegex.match(descriptor)
        if parsed is None:
            raise ValueError(f"Couldn't parse descriptor `{descriptor}` with pattern {DsetDescriptorRegex.pattern}")
        parsed_dict = parsed.groupdict()
    else:
        parsed_dict = {}

    # Filter `None` values
    parsed_ = {k: v for k, v in parsed_dict.items() if v is not None}
    config_ = {k: v for k, v in config_or_descriptor.items() if v is not None}

    # Create the final config
    final_config = {}
    for key in parsed_.keys() | config_.keys():
        parsed_value = parsed_.get(key, None)
        if parsed_value is not None and "+" in parsed_value:
            parsed_value = parsed_value.split("+")
        config_value = config_.get(key, None)
        final_config[key] = parsed_value or config_value

    final_config.pop("descriptor", None)
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
                    query.link = sections.sections[0].identifier

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
            queries.link = sections.identifier

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
    ) -> Iterable[DatasetConfig]:
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
