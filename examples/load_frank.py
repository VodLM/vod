from __future__ import annotations

import rich
import vod_configs
from vod_tools import arguantic

from src import vod_datasets


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    name: str = "frank_a"
    language: str = "en"
    split: str = "all"


def run(args: Args) -> None:
    """Showcase the `load_frank` function."""
    dset = vod_datasets.load_queries(
        vod_configs.QueriesDatasetConfig(
            name=args.name,
            subset=args.language,
            split=args.split,
        )
    )
    rich.print(dset)


if __name__ == "__main__":
    args = Args.parse()
    run(args)
