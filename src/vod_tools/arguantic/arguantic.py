from __future__ import annotations

import argparse

import pydantic
from typing_extensions import Self, Type


class Arguantic(pydantic.BaseModel):
    """Defines arguments using `pydantic` and parse them using `argparse`."""

    class Config:
        """Pydantic config."""

        extra = pydantic.Extra.forbid

    @classmethod
    def parse(cls: Type[Self]) -> Self:
        """Parse arguments using `argparse`."""
        parser = argparse.ArgumentParser()
        for name, field in cls.model_fields.items():
            parser.add_argument(f"--{name}", type=field.annotation or str, default=field.default)

        args = parser.parse_args()
        return cls(**vars(args))
