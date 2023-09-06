from __future__ import annotations

import string

import rich
import vod_configs
import vod_datasets
import vod_search
from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.theme import Theme
from vod_tools import arguantic


class QueryHighlighter(ReprHighlighter):
    """Apply style to match the query terms."""

    def __init__(self, query: str) -> None:
        super().__init__()
        # strip punctuation and replace with spaces
        query = query.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        keywords = query.split(" ")
        keywords = [f"({k})" for k in keywords if k not in STOP_WORDS and len(k)]
        self.highlights += [
            # match single and double quoted strings placed after `:`
            r"(?P<str_value>:\s*\"(.*?)\",\n|:\s*\'(.*?)\',\n)",
            rf"(?i)(?P<queryterm>{'|'.join(keywords)})",
        ]


THEME = Theme(
    {
        "repr.queryterm": "bold yellow",
        "repr.str_value": "white",
    }
)


class _Args(arguantic.Arguantic):
    name_or_path: str = "emozilla/quality"
    split: str = "validation[:100]"
    subsets: str = ""  # "high"


def run(args: _Args) -> None:
    """Run the script."""
    rich.print(args)

    # Load queries
    queries = vod_datasets.load_dataset(
        vod_configs.QueriesDatasetConfig(
            identifier="queries",
            name_or_path=args.name_or_path,
            split=args.split,
            subsets=args.subsets or None,
        )  # type: ignore
    )
    rich.print(queries)
    rich.print(queries[0])

    # Load sections
    sections = vod_datasets.load_dataset(
        vod_configs.SectionsDatasetConfig(
            identifier="sections",
            name_or_path=args.name_or_path,
            split=args.split,
            subsets=args.subsets or None,
            options=vod_configs.DatasetOptions(
                sectioning=vod_configs.FixedLengthSectioningConfig(
                    mode="fixed_length",
                    section_template=r"{{ title }} {{ content }}",
                    tokenizer_name_or_path="t5-base",
                    max_length=200,
                    stride=150,
                )
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

        console = Console(highlighter=QueryHighlighter(batch["query"][0]), theme=THEME)

        # Play with the query
        # batch["query"] = [""]
        batch["retrieval_ids"] = None  # batch["subset_ids"]
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
                        "subset_id": results[j]["subset_id"],
                        "score": sr.scores[0, j],
                        "_id": sr.indices[0, j],
                    }
                    for j in range(len(results))
                ],
            }
        )


STOP_WORDS = [
    "i",
    "s",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
]

if __name__ == "__main__":
    args = _Args.parse()
    run(args)
