import functools
import typing

import datasets
from vod_datasets.rosetta.adapters import SquadQueryWithContextsAdapter
from vod_datasets.rosetta.preprocessing import isolate_qa_and_sections
from vod_tools import fingerprint


def _re_inject_content(
    row: dict[str, typing.Any],
    sections_lookup: dict[str, dict[str, typing.Any]],
) -> dict[str, typing.Any]:
    subset_ids = row["subset_ids"]
    assert len(subset_ids) == 1, f"Expected 1 section, got {len(subset_ids)}"  # noqa: S101
    section = sections_lookup[subset_ids[0]]
    return {
        "contexts": [section["content"]],
        "titles": [section["title"]],
    }


def test_isolate_qa_and_sections(num_proc: int = 1) -> None:
    """Test isolating queries and sections."""
    datasets.disable_caching()
    squad_subset = datasets.load_dataset("squad_v2", split="validation[40:60]")  # type: ignore
    squad_subset: datasets.Dataset = SquadQueryWithContextsAdapter.translate(squad_subset)
    queries_and_sections = isolate_qa_and_sections(squad_subset, num_proc=num_proc)

    # Now process the section separately using a larger subset
    squad_superset = datasets.load_dataset("squad_v2", split="validation[:100]")  # type: ignore
    squad_superset: datasets.Dataset = SquadQueryWithContextsAdapter.translate(squad_superset)
    queries_and_sections_superset = isolate_qa_and_sections(squad_superset, num_proc=num_proc)
    sections_supserset_lookup = {
        queries_and_sections_superset.sections[j]["subset_id"]: queries_and_sections_superset.sections[j]
        for j in range(len(queries_and_sections_superset.sections))
    }

    # Re-inject the sections into the queries
    squad_rejoined: datasets.Dataset = queries_and_sections.queries.map(
        functools.partial(_re_inject_content, sections_lookup=sections_supserset_lookup),  # type: ignore
        num_proc=num_proc,
    )

    # Check the two datasets are the same
    assert len(squad_subset) == len(
        squad_rejoined
    ), f"Lengths do not match: {len(squad_subset)} != {len(squad_rejoined)}"
    for i in range(len(squad_subset)):
        row_ori = squad_subset[i]
        row_new = squad_rejoined[i]
        if set(row_ori.keys()) != set(row_new.keys()):
            raise ValueError(f"Row keys do not match: {row_ori.keys()} != {row_new.keys()}")
        for key in set(row_ori.keys()) - {"subset_ids"}:
            v_or = fingerprint.Hasher.hash(row_ori[key])
            v_new = fingerprint.Hasher.hash(row_new[key])
            assert v_or == v_new, f"Key=`{key}`. Row values do not match: {row_ori[key]} != {row_new[key]}"
