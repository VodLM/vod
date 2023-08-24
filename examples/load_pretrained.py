import argparse
import os
import pathlib
import shutil

import fsspec
import gcsfs
import pydantic
import rich
import torch
import transformers
from fsspec import callbacks
from typing_extensions import Self, Type


def init_gcloud_filesystem() -> fsspec.AbstractFileSystem:
    """Initialize a GCS filesystem."""
    try:
        token = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError as exc:
        raise RuntimeError("Missing `GOOGLE_APPLICATION_CREDENTIALS` environment variables. ") from exc
    try:
        project = os.environ["GCLOUD_PROJECT_ID"]
    except KeyError as exc:
        raise RuntimeError("Missing `GCLOUD_PROJECT_ID` environment variables. ") from exc
    return gcsfs.GCSFileSystem(token=token, project=project)


class Args(pydantic.BaseModel):
    """Parse args."""

    remote_dir: str = "raffle-models/research/"
    name: str = "mt5-large-B-v0"
    cache_dir: str = "~/.cache/vod/models/"
    delete_cache: bool = False

    @classmethod
    def parse(cls: Type[Self]) -> Self:
        """Parse arguments using `argparse`."""
        parser = argparse.ArgumentParser()
        for name, field in cls.model_fields.items():
            parser.add_argument(f"--{name}", type=field.annotation or str, default=field.default)

        args = parser.parse_args()
        return cls(**vars(args))

    @pydantic.field_validator("cache_dir", mode="before")
    def _validate_cache_dir(cls, v: str) -> str:
        """Validate the cache directory."""
        return pathlib.Path(v).expanduser().resolve().as_posix()


def run(args: Args) -> None:
    """Load a VOD model from GCS."""
    rich.print(args)

    # Download model
    local_path = pathlib.Path(args.cache_dir) / args.name
    fs = init_gcloud_filesystem()
    if args.delete_cache and local_path.exists():
        shutil.rmtree(local_path)

    if not local_path.exists():
        fs.get(
            str(pathlib.Path(args.remote_dir, args.name)),
            str(pathlib.Path(args.cache_dir)),
            recursive=True,
            callback=callbacks.TqdmCallback(),  # type: ignore
        )

    # Load model
    model = transformers.AutoModel.from_pretrained(local_path, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_path)
    rich.print(f"Model: {type(model)}")
    rich.print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    rich.print(f"Parameters: top-level keys: { {k.split('.')[0] for k in model.state_dict()} }")
    rich.print(tokenizer)

    # Test the model on dummy examples
    queries = [
        "[Lang=en] Q: What is the capital of France?",
        "[Lang=fr] Q: Quelle est la capitale de la France?",
        "[Lang=en] Q: What is computer science?",
        "[Lang=fr] Q: Qu'est-ce que l'informatique?",
        "[Lang=en] Q: What are your opening hours?",
        "[Lang=fr] Q: Quels sont vos horaires d'ouverture?",
    ]
    labels = torch.tensor(
        [
            0,
            1,
            3,
            4,
            6,
        ]
    )
    mask = torch.tensor(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    sections = [
        "[Lang=en] Title: Paris| D: Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles).",  # noqa: E501
        "[Lang=fr] Title: Paris| D: Paris est la capitale de la France et le chef-lieu de la région d’Île-de-France. La ville se situe au cœur d’un vaste bassin sédimentaire aux sols fertiles et au climat tempéré, le bassin parisien, sur une boucle de la Seine, entre les confluents de celle-ci avec la Marne et l’Oise.",  # noqa: E501
        "[Lang=en] Title: Computer science| D: Computer science is the study of algorithmic processes, computational machines and computation itself.",  # noqa: E501
        "[Lang=fr] Title: Informatique| D: L’informatique est un domaine d’activité scientifique, technique et industriel concernant le traitement automatique de l’information par l’exécution de programmes informatiques par des machines : des systèmes embarqués, des ordinateurs, des robots, des automates, etc.",  # noqa: E501
        "[Lang=en] Title: Opening hours| D: Opening hours are the times during which an establishment such as a store, business, or school is open.",  # noqa: E501
        "[Lang=fr] Title: Horaires d'ouverture| D: Les horaires d'ouverture sont les heures pendant lesquelles un établissement tel qu'un magasin, une entreprise ou une école est ouvert.",  # noqa: E501
        "[Lang=en] Title: France| D: France, officially the French Republic, is a country whose territory consists of metropolitan France in Western Europe and several overseas regions and territories.",  # noqa: E501
        "[Lang=fr] Title: France| D: La France, en forme longue depuis 1875 la République française, est un État souverain transcontinental dont le territoire métropolitain est situé en Europe de l’Ouest.",  # noqa: E501
    ]
    labels = torch.tensor(
        [
            0,
            1,
            2,
            3,
            4,
            6,
        ]
    )
    mask = torch.tensor(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=torch.bool,
    )

    encoded_queries = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
    encoded_sections = tokenizer(sections, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        hq = model(**encoded_queries).pooler_output
        hs = model(**encoded_sections).pooler_output

    sim = torch.einsum("qd,sd->qs", hq, hs)
    rich.print(sim)
    sim = torch.where(mask, sim, torch.full_like(sim, float("-inf")))
    rich.print(
        {
            "labels": labels,
            "predictions": sim.argmax(dim=1),
        }
    )


if __name__ == "__main__":
    args = Args.parse()
    run(args)
