import argparse

import pydantic


class Arguantic(pydantic.BaseModel):
    """Defines arguments using `pydantic` and parse them using `argparse`."""

    class Config:
        """Pydantic config"""

        extra = pydantic.Extra.forbid

    @classmethod
    def parse(cls) -> "Arguantic":
        """Parse arguments using `argparse`"""
        parser = argparse.ArgumentParser()
        for field in cls.__fields__.values():
            parser.add_argument(f"--{field.name}", type=field.type_, default=field.default)

        args = parser.parse_args()
        return cls(**vars(args))
