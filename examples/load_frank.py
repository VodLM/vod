from __future__ import annotations

import collections
import sys

import rich
from raffle_ds_research.tools import arguantic, raffle_datasets
from rich.progress import track


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


FRANK_A_KBIDS = [
    4,
    9,
    10,
    14,
    20,
    26,
    29,
    30,
    32,
    76,
    96,
    105,
    109,
    130,
    156,
    173,
    188,
    195,
    230,
    242,
    294,
    331,
    332,
    541,
    598,
    1061,
    1130,
    1148,
    1242,
    1264,
    1486,
    1599,
    1663,
    1665,
]
FRANK_B_KBIDS = [
    2,
    6,
    7,
    11,
    12,
    15,
    24,
    25,
    33,
    35,
    37,
    80,
    81,
    121,
    148,
    194,
    198,
    269,
    294,
    334,
    425,
    554,
    596,
    723,
    790,
    792,
    1284,
    1584,
    1589,
]

if __name__ == "__main__":
    args = Args.parse()

    frank = raffle_datasets.load_frank(
        language=args.language,
        subset_name=args.split,
        version=args.version,
        only_positive_sections=False,
        invalidate_cache=bool(args.invalidate_cache),
    )

    kb_ids = {frank.sections[j]["kb_id"] for j in range(len(frank.sections))}
    rich.print(sorted(kb_ids))
    kb_ids = {frank.qa_splits["train"][j]["kb_id"] for j in range(len(frank.qa_splits["train"]))}
    rich.print(sorted(kb_ids))
    sys.exit()

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
