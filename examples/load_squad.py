from __future__ import annotations

import rich
from vod_tools import arguantic, vod_data


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    language: str = "en"
    sample_idx: int = 101


if __name__ == "__main__":
    args = Args.parse()
    squad = vod_data.load_squad(language=args.language)
    rich.print(squad)

    rich.print("==== train ====")
    question = squad.qa_splits["train"][args.sample_idx]
    section = squad.sections[question["section_ids"][0]]
    rich.print({"question": question, "section": section})
