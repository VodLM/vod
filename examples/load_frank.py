from __future__ import annotations

import collections

import rich
from rich.progress import track

from raffle_ds_research.tools import arguantic, raffle_datasets


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    language: str = "en"
    split: str = "A"
    version: int = 0
    invalidate_cache: int = 1
    sample_idx: int = 42


def _split_text(x: str) -> list[str]:
    """Tokenize a text and returns the tokens."""
    return x.split()


if __name__ == "__main__":
    args = Args.parse()

    frank = raffle_datasets.load_frank(
        language=args.language,
        subset_name=args.split,
        version=args.version,
        only_positive_sections=False,
        invalidate_cache=bool(args.invalidate_cache),
    )
    rich.print({"frank_full": frank})
    n_sections_full = len(frank.sections)
    rich.print({f"training-question-{args.sample_idx}": frank.qa_splits["train"][0]})

    n_tokens = []
    for j in track(range(len(frank.sections)), description="Counting tokens"):
        sec = frank.sections[j]
        n_tokens.append(len(_split_text(sec["content"])))

    rich.print(
        f"> Average number of tokens per section: "
        f"{sum(n_tokens) / len(n_tokens):.2f}\n"
        f"> Max number of tokens per section: {max(n_tokens)}\n"
        f"> Min number of tokens per section: {min(n_tokens)}\n"
    )

    counts = collections.Counter(n_tokens)
    rich.print("> Counts:")
    rich.print(dict(sorted(counts.items(), key=lambda x: x[0])))
