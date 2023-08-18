import pydantic


class StrictModel(pydantic.BaseModel):
    """A pydantic model with strict configuration."""

    class Config:
        """Pydantic configuration."""

        extra = "ignore"
        frozen = False  # before named "frozen=True"
        from_attributes = True
