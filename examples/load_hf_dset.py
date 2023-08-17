from __future__ import annotations

import rich
import vod_configs
from vod_tools import arguantic

from src import vod_datasets

DATASET_CONFIGS = {
    "ms_marco": {
        "name": "ms_marco",
        "subsets": ["v2.1"],
        "splits": ["test"],
    },
    "mmlu": {
        "name": "cais/mmlu",
        "subsets": ["astronomy", "prehistory"],
        "splits": ["dev"],
    },
    "squad": {
        "name": "squad_v2",
        "splits": ["validation"],
    },
    "quality": {
        "name": "emozilla/quality",
    },
    "nq_open": {
        "name": "nq_open",
        "splits": ["validation"],
    },
    "trivia_qa": {
        "name": "trivia_qa",
        "subsets": ["rc.wikipedia"],
        "splits": ["test"],
    },
}


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    config_name: str = "trivia_qa"


def run(args: Args) -> None:
    """Showcase the `load_frank` function."""
    config = DATASET_CONFIGS.get(args.config_name)

    if not config:
        rich.print(f"Configuration for {args.config_name} not found!")
        return

    dset = vod_datasets.load_queries(
        vod_configs.BaseDatasetConfig(
            name=config.get("name"),
            subsets=config.get("subsets", []),
            splits=config.get("splits", []),
        )
    )
    rich.print(dset)


if __name__ == "__main__":
    args = Args.parse()
    run(args)
