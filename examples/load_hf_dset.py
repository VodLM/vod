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
        "link": "wikipedia.20200501.en:validation",
        "splits": ["validation"],
    },
    "trivia_qa": {
        "name": "trivia_qa",
        "subsets": ["rc.wikipedia"],
        "splits": ["test"],
    },
    "wiki": {
        "name": "wikipedia",
        "subsets": ["20200501.en"],
        "splits": ["validation"],
        # "fraction": "0.1%",
    },
}


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    name: str = "squad"


def run(args: Args) -> None:
    """Showcase the `load_frank` function."""
    try:
        config = DATASET_CONFIGS[args.name]
    except KeyError as exc:
        raise KeyError(f"Configuration for `{args.name}` not found!") from exc

    dset = vod_datasets.load_queries(vod_configs.BaseDatasetConfig(**config))
    rich.print(dset)

    dset = vod_datasets.load_sections(vod_configs.BaseDatasetConfig(**config))
    rich.print(dset)


if __name__ == "__main__":
    args = Args.parse()
    run(args)
