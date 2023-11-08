import string

from rich.highlighter import ReprHighlighter
from rich.theme import Theme


class QueryHighlighter(ReprHighlighter):
    """Apply style to match the query terms."""

    def __init__(
        self,
        query: str,
        subset_ids: None | list[str] = None,
        retrieval_ids: None | list[str] = None,
    ) -> None:
        super().__init__()
        # strip punctuation and replace with spaces
        query = query.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        keywords = query.split(" ")
        keywords = [f"({k})" for k in keywords if k not in STOP_WORDS and len(k)]
        self.highlights += [
            # match single and double quoted strings placed after `:`
            r"(?P<str_value>:\s*\"(.*?)\",\n|:\s*\'(.*?)\',\n)",
            rf"(?i)(?P<query_term>{'|'.join(keywords)})",
        ]
        if subset_ids:
            self.highlights.append(rf"(?P<subset_id>{'|'.join(subset_ids)})")
        if retrieval_ids:
            self.highlights.append(rf"(?P<retrieval_id>{'|'.join(retrieval_ids)})")


QUERY_HIGHLIGHTER_THEME = Theme(
    {
        "repr.query_term": "bold yellow",
        "repr.subset_id": "bold green",
        "repr.retrieval_id": "bold red",
        "repr.str_value": "white",
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
