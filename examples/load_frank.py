import rich

from raffle_ds_research.tools import arguantic, raffle_datasets


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    language: str = "en"
    split: str = "A"
    version: int = 0
    invalidate_cache: int = 1
    sample_idx: int = 42


if __name__ == "__main__":
    args = Args.parse()

    frank = raffle_datasets.load_frank(
        language=args.language,
        subset_name=args.split,
        version=args.version,
        only_positive_sections=True,
        invalidate_cache=bool(args.invalidate_cache),
    )
    rich.print(dict(frank_only_positives=frank))
    rich.print(frank.qa_splits["train"][0])
    n_sections_small = len(frank.sections)

    frank = raffle_datasets.load_frank(
        language=args.language,
        subset_name=args.split,
        version=args.version,
        only_positive_sections=False,
        invalidate_cache=bool(args.invalidate_cache),
    )
    rich.print(dict(frank_full=frank))
    n_sections_full = len(frank.sections)

    rich.print(f"Fraction of sections in the small version: {n_sections_small / n_sections_full:.2%}")

    for split, questions in frank.qa_splits.items():
        n_questions_with_single_section = sum(len(question["section_ids"]) == 1 for question in questions)
        rich.print(
            f"   {split}: {n_questions_with_single_section / len(questions):.2%} questions with a single section"
        )

    rich.print({f"training-question-{args.sample_idx}": frank.qa_splits["train"][0]})
