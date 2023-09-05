from __future__ import annotations

import rich
from vod_tools import arguantic

from src import (
    vod_configs,
    vod_datasets,
    vod_search,
)


class _Args(arguantic.Arguantic):
    name_or_path: str = "squad_v2"
    split: str = "validation[:20]"
    cleanup_cache: bool = False


def run(args: _Args) -> None:
    """Run the script."""
    rich.print(args)
    # datasets.logging.set_verbosity_debug()

    # Load queries
    queries = vod_datasets.load_dataset(
        vod_configs.QueriesDatasetConfig(
            identifier="queries",
            name_or_path=args.name_or_path,
            split=args.split,
        )  # type: ignore
    )
    if args.cleanup_cache:
        queries.cleanup_cache_files()
    rich.print(queries)
    rich.print(queries[0])

    # Load sections
    sections = vod_datasets.load_dataset(
        vod_configs.SectionsDatasetConfig(
            identifier="sections",
            name_or_path=args.name_or_path,
            split=args.split,  # <--- load all sections
        )  # type: ignore
    )
    if args.cleanup_cache:
        sections.cleanup_cache_files()
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

        # Play with the query
        # batch["query"] = ["Normans"]
        # batch["retrieval_ids"] = None # batch["subset_ids"]
        # batch["subset_ids"] = None

        sr = client.search(
            text=batch["query"],
            subset_ids=batch.get("subset_ids", None),
            ids=batch.get("retrieval_ids", None),
            top_k=10,
        )
        rich.print(sr)
        results = [sections[int(i)] for i in sr.indices[0]]
        rich.print(
            {
                "query": {
                    "query": batch["query"],
                    "subset_ids": batch.get("subset_ids", None),
                    "retrieval_ids": batch.get("retrieval_ids", None),
                },
                "results": [
                    {
                        "title": results[j]["title"],
                        "content": results[j]["content"],
                        "id": results[j]["id"],
                        "subset_id": results[j]["subset_id"],
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
