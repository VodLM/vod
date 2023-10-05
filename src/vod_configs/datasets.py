import typing as typ

import datasets
import pydantic
from datasets import fingerprint
from typing_extensions import Self, Type
from vod_tools.misc.config import as_pyobj_validator

from .search import (
    HybridSearchFactoryConfig,
)
from .support import SectioningConfig
from .utils.base import StrictModel


@typ.runtime_checkable
class DatasetLoader(typ.Protocol):
    """A dataset loader."""

    def __call__(self, subset: None | str = None, split: None | str = None, **kws: typ.Any) -> datasets.Dataset:
        """Load a dataset."""
        ...


class DatasetOptionsDiff(StrictModel):
    """Preprocessing options diff."""

    prep_map_kws: None | dict[str, typ.Any] = None
    subset_size: None | int = None
    sectioning: None | SectioningConfig = None

    # validators
    _validate_prep_map_kws = pydantic.field_validator("prep_map_kws", mode="before")(as_pyobj_validator)


class DatasetOptions(StrictModel):
    """Preprocessing options."""

    prep_map_kws: dict[str, typ.Any] = pydantic.Field(
        default_factory=dict,
        description="Kwargs for `datasets.map(...)`.",
    )
    subset_size: None | int = pydantic.Field(
        default=None,
        description="Take a subset of the dataset.",
    )
    sectioning: None | SectioningConfig = pydantic.Field(
        default=None,
        description="Configures a sectioning algorithm to split input documents/sections into smaller chunks.",
    )

    # validators
    _validate_prep_map_kws = pydantic.field_validator("prep_map_kws", mode="before")(as_pyobj_validator)

    def __add__(self: Self, other: None | DatasetOptionsDiff) -> Self:
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
    name_or_path: str | DatasetLoader = pydantic.Field(
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

    def __hash__(self) -> int:
        """Hash the object based on its name and split."""
        return hash(self.fingerprint())

    def fingerprint(self) -> str:
        """Return the hexdigest of the hash."""
        data = self.model_dump()
        if not isinstance(data["name_or_path"], str):
            data["name_or_path"] = fingerprint.Hasher.hash(data["name_or_path"])
        return fingerprint.Hasher.hash(data)

    @pydantic.field_validator("options", mode="before")
    @classmethod
    def _validate_options(
        cls: Type[Self], v: None | dict[str, typ.Any] | DatasetOptions
    ) -> dict[str, typ.Any] | DatasetOptions:
        if isinstance(v, DatasetOptions):
            return v

        return DatasetOptions(**(v or {}))

    @pydantic.field_validator("subsets", mode="before")
    @classmethod
    def _validate_subsets(cls: Type[Self], value: None | str | list[str]) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]

        return [str(x) for x in value]


class QueriesDatasetConfig(BaseDatasetConfig):
    """Defines a query dataset."""

    field: typ.Literal["query"] = "query"
    link: None | str = pydantic.Field(
        None,
        description="Identifier of the `Sections` dataset to search into.",
    )


class SectionsDatasetConfig(BaseDatasetConfig):
    """Defines a section dataset."""

    field: typ.Literal["section"] = "section"
    search: None | HybridSearchFactoryConfig = pydantic.Field(
        None,
        description="Search config diffs for this dataset",
    )

    @pydantic.field_validator("search", mode="before")
    @classmethod
    def _validate_search(cls: typ.Type[Self], v: None | dict[str, typ.Any]) -> None | dict[str, typ.Any] | Self:
        if v is None or isinstance(v, HybridSearchFactoryConfig):
            return v

        return {k: v[k] for k in v}


DatasetConfig = QueriesDatasetConfig | SectionsDatasetConfig
