from __future__ import annotations

import rich
import vod_configs
import vod_datasets
import vod_search
from rich.console import Console
from vod_tools import arguantic

from examples.search.utils import (
    QUERY_HIGHLIGHTER_THEME,
    QueryHighlighter,
)

DATASET_CONFIGS = {
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
    "mmarco-french": {
        "identifier": "mmarco-french",
        "name_or_path": [
            vod_datasets.BeirDatasetLoader(what="queries"),
            vod_datasets.BeirDatasetLoader(what="sections"),
        ],
        "subsets": "mmarco/french",
        "split": "train",
    },
    "scifact": {
        "identifier": "scifact",
        "name_or_path": [
            vod_datasets.BeirDatasetLoader(what="queries"),
            vod_datasets.BeirDatasetLoader(what="sections"),
        ],
        "subsets": "scifact",
        "split": "train",
    },
}


class _Args(arguantic.Arguantic):
    name: str = "quality"
    use_sectioning: int = 1


def run(args: _Args) -> None:
    """Run the script."""
    rich.print(args)
    dataset_config = DATASET_CONFIGS[args.name]
    name_or_path = dataset_config.pop("name_or_path")
    if isinstance(name_or_path, list):
        queries_name_or_path, sections_name_or_path = name_or_path
    else:
        queries_name_or_path = sections_name_or_path = name_or_path

    # Load queries
    queries = vod_datasets.load_dataset(
        vod_configs.QueriesDatasetConfig(
            name_or_path=queries_name_or_path,
            **dataset_config,
        )  # type: ignore
    )
    rich.print(queries)
    rich.print(queries[0])

    # Sectioning configuration
    sectionizer = (
        vod_configs.FixedLengthSectioningConfig(
            section_template=r"{{ title }} {{ content }}",
            tokenizer_name_or_path="t5-base",
            max_length=200,
            stride=150,
        )
        if args.use_sectioning
        else None
    )

    # Load sections
    sections = vod_datasets.load_dataset(
        vod_configs.SectionsDatasetConfig(
            name_or_path=sections_name_or_path,
            **dataset_config,
            options=vod_configs.DatasetOptions(
                sectioning=sectionizer,
            ),  # type: ignore
        )
    )
    rich.print(sections)
    rich.print(sections[0])

    # Build the search engine
    with vod_search.build_elasticsearch_index(
        sections=sections,  # type: ignore
        config=vod_configs.ElasticsearchFactoryConfig(
            section_template=r"{{ title }} {{ content }}",
        ),  # type: ignore
    ) as master:
        client = master.get_client()
        rich.print(client)

        batch = queries[:1]

        # Make a rich console highlighter
        console = Console(
            highlighter=QueryHighlighter(
                batch["query"][0],
                subset_ids=batch.get("subset_ids", [[]])[0],
                retrieval_ids=batch.get("retrieval_ids", [[]])[0],
            ),
            theme=QUERY_HIGHLIGHTER_THEME,
        )

        # Play with the query
        # batch["query"] = [""]
        # batch["retrieval_ids"] = None  # batch["subset_ids"]
        # batch["subset_ids"] = None

        sr = client.search(
            text=batch["query"],
            subset_ids=batch.get("subset_ids", None),
            ids=batch.get("retrieval_ids", None),
            top_k=10,
        )
        rich.print(sr)
        results = [sections[int(i)] for i in sr.indices[0]]
        console.print(
            {
                "query": {
                    "query": batch["query"],
                    "subset_ids": batch.get("subset_ids", None),
                    "retrieval_ids": batch.get("retrieval_ids", None),
                    "answers": batch.get("answers", None),
                    "answer_scores": batch.get("answer_scores", None),
                },
                "results": [
                    {
                        "title": results[j]["title"],
                        "content": results[j]["content"],
                        "id": results[j]["id"],
                        "subset_id": results[j].get("subset_id", None),
                        "score": sr.scores[0, j],
                        "_id": sr.indices[0, j],
                    }
                    for j in range(len(results))
                ],
            }
        )


if __name__ == "__main__":
    args = _Args.parse()
    run(args)
