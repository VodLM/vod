import pydantic
from typing_extensions import Self, Type
from vod_datasets.rosetta.adapters import aliases


class WithContexstMixin(pydantic.BaseModel):
    """A mixin for models with context."""

    titles: None | list[str] = pydantic.Field(
        None,
        validation_alias=aliases.TITLES_ALIASES,
    )
    contexts: list[str] = pydantic.Field(
        ...,
        validation_alias=aliases.CONTEXTS_ALIASES,
    )

    # Validators
    @pydantic.field_validator("titles", mode="before")
    @classmethod
    def _validate_titles(cls: Type[Self], v: None | str | list[str]) -> None | list[str]:
        """Validate titles."""
        if isinstance(v, str):
            return [v]
        return v

    @pydantic.field_validator("contexts", mode="before")
    @classmethod
    def _validate_contexts(cls: Type[Self], v: str | list[str]) -> list[str]:
        """Validate contexts."""
        if isinstance(v, str):
            return [v]
        return v
