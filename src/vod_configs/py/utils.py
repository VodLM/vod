import typing

import pydantic


class StrictModel(pydantic.BaseModel):
    """A pydantic model with strict configuration."""

    class Config:
        """Pydantic configuration."""

        extra = "ignore"
        frozen = False  # before named "frozen=True"
        from_attributes = True


class AllowMutations:
    """Temporarily allow mutations on a model."""

    def __init__(self, model: pydantic.BaseModel) -> None:
        self.model = model
        self._is_frozen = model.model_config.get("frozen", None)

    def __enter__(self) -> pydantic.BaseModel:
        self.model.model_config["frozen"] = False
        return self.model

    def __exit__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        if self._is_frozen is None:
            self.model.model_config.pop("frozen", None)
        else:
            self.model.model_config["frozen"] = self._is_frozen
