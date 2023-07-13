import pydantic


class StrictModel(pydantic.BaseModel):
    """A pydantic model with strict configuration."""

    class Config:
        """Pydantic configuration."""

        extra = "forbid"
        allow_mutation = False
