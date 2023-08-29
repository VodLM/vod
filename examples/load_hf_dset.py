from __future__ import annotations

import rich
import vod_configs
from vod_tools import arguantic

from src import vod_datasets

DATASET_CONFIGS = {
    "msmarco": {
        "identifier": "msmarco",
        "name_or_path": "ms_marco",
        "subsets": ["v2.1"],
        "split": "test",
    },
    "mmlu": {
        "identifier": "mmlu",
        "name_or_path": "cais/mmlu",
        "subsets": ["astronomy", "prehistory"],
        "split": "dev",
    },
    "squad": {
        "identifier": "squad",
        "name_or_path": "squad_v2",
        "split": "validation",
    },
    "quality": {
        "identifier": "quality",
        "name_or_path": "emozilla/quality",
    },
    "race": {
        "identifier": "race",
        "name_or_path": "race",
        "subset": "high",
        "split": "test",
    },
    "nqopen": {
        "identifier": "nqopen",
        "name_or_path": "nq_open",
        "split": "validation",
    },
    "trivia_qa": {
        "identifier": "trivia",
        "name_or_path": "trivia_qa",
        "subsets": ["rc.wikipedia"],
        "split": "test",
    },
    "wiki": {
        "identifier": "wiki",
        "name_or_path": "wikipedia",
        "subsets": ["20220301.simple"],
        "split": "train",
    },
    "flan": {
        "identifier": "flan",
        "name_or_path": "Muennighoff/flan",
        "split": "test",
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
    rich.print(dset[0])

    dset = vod_datasets.load_sections(vod_configs.BaseDatasetConfig(**config))
    rich.print(dset)
    rich.print(dset[0])


if __name__ == "__main__":
    args = Args.parse()
    run(args)
