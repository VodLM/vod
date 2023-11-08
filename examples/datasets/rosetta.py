from __future__ import annotations

import typing

import datasets
import rich
import vod_configs
import vod_datasets
from vod_tools import arguantic


def my_loader(
    subset: str | None = None,  # noqa: ARG001
    split: str | None = None,  # noqa: ARG001
    **kws: dict[str, typing.Any],
) -> datasets.Dataset:
    """Define a custom data loader."""
    return datasets.Dataset.from_list(
        [
            {
                "question": "What is the meaning of life?",
                "answer": "42",
            },
        ]
    )


DATASET_CONFIGS = {
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
    "my_data": {
        "identifier": "my_data",
        "name_or_path": my_loader,
    },
    "beir-scifact": {
        "identifier": "beir-scifact",
        "name_or_path": [
            vod_datasets.BeirDatasetLoader(what="queries"),
            vod_datasets.BeirDatasetLoader(what="sections"),
        ],
        "subsets": "scifact",
        "split": "train",
    },
    "beir-mmarco": {
        "identifier": "beir-mmarco",
        "name_or_path": [
            vod_datasets.BeirDatasetLoader(what="queries"),
            vod_datasets.BeirDatasetLoader(what="sections"),
        ],
        "subsets": "mmarco/french",
        "split": "train",
    },
    "msmarco": {
        "identifier": "beir-msmarco",
        "name_or_path": [
            vod_datasets.BeirDatasetLoader(what="queries"),
            vod_datasets.BeirDatasetLoader(what="sections"),
        ],
        "subsets": "hf://BeIR/msmarco",
        "split": "train",
    },
}


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    name: str = "squad"


def run(args: Args) -> None:
    """Showcase the `load_dataset` function."""
    try:
        config = DATASET_CONFIGS[args.name]
    except KeyError as exc:
        raise KeyError(f"Configuration for `{args.name}` not found!") from exc

    dset = vod_datasets.load_dataset(
        vod_configs.QueriesDatasetConfig(
            **{
                **config,
                "name_or_path": config["name_or_path"][0]
                if isinstance(config["name_or_path"], list)
                else config["name_or_path"],
            }
        ),
    )
    rich.print(dset)
    rich.print(dset[0])

    dset = vod_datasets.load_dataset(
        vod_configs.SectionsDatasetConfig(
            **{
                **config,
                "name_or_path": config["name_or_path"][1]
                if isinstance(config["name_or_path"], list)
                else config["name_or_path"],
            }
        )
    )
    rich.print(dset)
    rich.print(dset[0])


if __name__ == "__main__":
    args = Args.parse()
    run(args)
