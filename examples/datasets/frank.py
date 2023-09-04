from __future__ import annotations

import rich
import vod_configs
from vod_tools import arguantic

from src import vod_datasets


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    frank_split: str = "A"
    language: str = "en"
    split: str = "all"


def run(args: Args) -> None:
    """Showcase the `load_frank` function."""
    dset = vod_datasets.load_queries(
        vod_configs.QueriesDatasetConfig(
            identifier=f"frank-{args.frank_split}-queries-{args.language}",
            name_or_path=vod_datasets.FrankDatasetLoader(
                frank_split=args.frank_split,
                what="queries",
            ),
            subsets=[args.language],
            split=args.split,
            link=None,
            options=vod_configs.DatasetOptions(),
        )
    )
    rich.print(dset)


if __name__ == "__main__":
    args = Args.parse()
    run(args)
