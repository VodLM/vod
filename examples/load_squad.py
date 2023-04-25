import rich

from raffle_ds_research.tools import arguantic, raffle_datasets


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    language: str = "en"
    sample_idx: int = 101


if __name__ == "__main__":
    args = Args.parse()
    squad = raffle_datasets.load_squad(language=args.language)
    rich.print(squad)

    rich.print("==== train ====")
    question = squad.qa_splits["train"][args.sample_idx]
    section = squad.sections[question["section_ids"][0]]
    rich.print(dict(question=question, section=section))
