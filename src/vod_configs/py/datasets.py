from __future__ import annotations

import copy
import itertools
import re
from typing import Any, Iterable, Literal, Optional, Protocol, TypeVar, Union, runtime_checkable

import datasets
import omegaconf
import pydantic
from datasets import fingerprint
from hydra.utils import instantiate
from loguru import logger
from typing_extensions import Self, Type
from vod_tools.misc.config import as_pyobj_validator

from .search import HybridSearchFactoryConfig, MutliSearchFactoryDiff, SearchFactoryDefaults
from .sectioning import SectioningConfig
from .utils import AllowMutations, StrictModel

_CONFIG_EXPAND_KEY = "__vars__"


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

    identifier: pydantic.constr(to_lower=True) = pydantic.Field(  # type: ignore | auto-lowercase
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
        if isinstance(self.name_or_path, str):
            desc = self.name_or_path
        else:
            clsn_ame = type(self.name_or_path).__name__
            cls_hash = fingerprint.Hasher.hash(self.name_or_path)
            desc = f"{clsn_ame}({cls_hash})"
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
        config: dict | omegaconf.DictConfig,
        *,
        base_options: Optional[DatasetOptions] = None,
        base_search: Optional[HybridSearchFactoryConfig] = None,
    ) -> Self:
        """Parse a config dictionary or dataset name into a structured config."""
        # parse `options` if provided
        if isinstance(config, omegaconf.DictConfig):
            config: dict[str, Any] = omegaconf.OmegaConf.to_container(config, resolve=True)  # type: ignore

        if isinstance(config["name_or_path"], dict):
            config["name_or_path"] = instantiate(config["name_or_path"])

        base_options = base_options or DatasetOptions()
        options = DatasetOptionsDiff(**(config["options"] if "options" in config else {}))
        config["options"] = base_options + options

        # parse `search` if provided, and combine with the base search config
        if base_search is not None:
            if "search" in config:
                base_search = base_search + MutliSearchFactoryDiff.parse(config["search"])
            config["search"] = base_search
        return cls(**config)  # type: ignore


class QueriesDatasetConfig(BaseDatasetConfig):
    """Defines a query dataset."""

    field: Literal["query"] = "query"
    link: Optional[str] = pydantic.Field(
        None,
        description="Identifier of the `Sections` dataset to search into.",
    )


class SectionsDatasetConfig(BaseDatasetConfig):
    """Defines a section dataset."""

    field: Literal["section"] = "section"
    search: Optional[HybridSearchFactoryConfig] = pydantic.Field(
        None,
        description="Search config diffs for this dataset",
    )


DatasetConfig = Union[QueriesDatasetConfig, SectionsDatasetConfig]


BDC = TypeVar("BDC", bound=BaseDatasetConfig)
Cv = TypeVar("Cv", bound=Union[str, dict, list])


def _expand_dynamic_configs(x: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand dynamic configurations (e.g. `__vars__`)."""
    expanded_x = []
    for y in x:
        if not isinstance(y, dict):
            expanded_x.append(y)
            continue
        variables = y.pop(_CONFIG_EXPAND_KEY, None)  # type: ignore
        if variables is None:
            expanded_x.append(y)
            continue

        # Take the combinations of the variables
        def _sub(v: Cv, target: str, value: Any) -> Cv:  # noqa: ANN401
            if isinstance(v, str):
                # replace `{target}` with `value`
                return re.sub(rf"\{{\s*{target}\s*\}}", str(value), v)
            if isinstance(v, dict):
                return {k: _sub(v, target, value) for k, v in v.items()}  # type: ignore
            if isinstance(v, list):
                return [_sub(v, target, value) for v in v]  # type: ignore
            return v

        keys = list(variables.keys())
        values = list(variables.values())
        for comb in itertools.product(*values):
            new_y = copy.deepcopy(y)
            for pat, val in zip(keys, comb):
                new_y = {k: _sub(v, pat, val) for k, v in new_y.items()}
            expanded_x.append(new_y)

    return expanded_x


def _omegaconf_to_dict_list(
    x: dict | omegaconf.DictConfig | list[dict] | omegaconf.ListConfig,
) -> list[dict]:
    if isinstance(x, (omegaconf.DictConfig, omegaconf.ListConfig)):
        x = omegaconf.OmegaConf.to_container(x, resolve=True)  # type: ignore

    if isinstance(x, dict):
        x = [x]

    return x  # type: ignore


def _parse_list_dset_configs(
    x: dict | omegaconf.DictConfig | list[dict] | omegaconf.ListConfig,
    cls: Type[BDC],
    **kwargs: Any,
) -> list[BDC]:
    x = _omegaconf_to_dict_list(x)
    # Resolve dynamic configurations (e.g. `__vars__`)
    x = _expand_dynamic_configs(x)  # type: ignore

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
        base_search: Optional[HybridSearchFactoryConfig] = None,
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
        base_search: Optional[HybridSearchFactoryConfig] = None,
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

    @pydantic.model_validator(mode="after")
    def _validate_links(self: Self) -> Self:
        """Check that the queries are pointing to valid sections."""
        section_ids = [s.identifier for s in self.sections.sections]
        linked_queries = {sid: [] for sid in section_ids}
        if len(linked_queries) != len(section_ids):
            raise ValueError(
                f"Duplicate section identifiers found: `{section_ids}`. "
                f"Make sure to assign each section dataset with a uniquer identifier."
            )

        # Assign queries to sections
        for query in self.queries.train + self.queries.val:
            if query.link not in section_ids:
                raise ValueError(
                    f"Query `{query.identifier}` points to invalid section ID `{query.link}`. "
                    f"Available section IDs: `{section_ids}`"
                )
            linked_queries[query.link].append(query.identifier)

        # Check that all sections have at least one query
        # Drop the sections that have no queries
        for sid in list(linked_queries.keys()):
            if not linked_queries[sid]:
                logger.warning(f"Section `{sid}` has no queries; dropping it.")
                self.sections.sections = [s for s in self.sections.sections if s.identifier != sid]

        return self


class BenchmarkDatasetConfig(StrictModel):
    """Defines a benchmark."""

    queries: QueriesDatasetConfig
    sections: SectionsDatasetConfig

    @classmethod
    def parse(
        cls: Type[Self],
        config: dict | omegaconf.DictConfig,
        base_options: Optional[DatasetOptions] = None,
        base_search: Optional[HybridSearchFactoryConfig] = None,
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

        # Resolve dynamic configurations (e.g. `__vars__`)
        benchmark_configs = _omegaconf_to_dict_list(config["benchmark"])
        benchmark_configs = _expand_dynamic_configs(benchmark_configs)  # type: ignore

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
                for cfg in benchmark_configs
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
