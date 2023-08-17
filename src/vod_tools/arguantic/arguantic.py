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
        for field_name, field in cls.__fields__.items():
            parser.add_argument(f"--{field_name}", type=field.annotation, default=field.default)

        args = parser.parse_args()
        return cls(**vars(args))
