import argparse

import pydantic


class Arguantic(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    @classmethod
    def parse(cls):
        parser = argparse.ArgumentParser()
        for field in cls.__fields__.values():
            parser.add_argument(f"--{field.name}", type=field.type_, default=field.default)

        args = parser.parse_args()
        return cls(**vars(args))
