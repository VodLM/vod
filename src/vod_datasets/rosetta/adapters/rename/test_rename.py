import datasets
from vod_datasets.rosetta import models, testing

from .adapter import RenameQueryAdapter, RenameSectionAdapter


def test_rename_dummy_queries() -> None:
    """Test parsing dummy queries."""
    dummy_queries = [
        {
            "question": "What is the meaning of life?",
            "answer": ["42"],
        },
    ]
    data = datasets.Dataset.from_list(dummy_queries)
    return testing.test_parse_dataset(
        data=data,
        adapter_cls=RenameQueryAdapter,
        output_model=models.QueryModel,
    )


def test_medwiki_as_sections() -> None:
    """Test parsing the MedWiki dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset(
            "VOD-LM/medwiki",
            split="train[:10]",
        ),  # type: ignore
        adapter_cls=RenameSectionAdapter,
        output_model=models.SectionModel,
    )


def test_agnews_as_queries() -> None:
    """Test parsing the AG News dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset(
            "ag_news",
            split="train[:10]",
        ),  # type: ignore
        adapter_cls=RenameQueryAdapter,
        output_model=models.QueryModel,
    )


def test_agnews_as_sections() -> None:
    """Test parsing the AG News dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset(
            "ag_news",
            split="train[:10]",
        ),  # type: ignore
        adapter_cls=RenameSectionAdapter,
        output_model=models.SectionModel,
    )


def test_nq_open_as_queries() -> None:
    """Test parsing the Natural Questions Open dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset("nq_open", split="validation[:10]"),  # type: ignore
        adapter_cls=RenameQueryAdapter,
        output_model=models.QueryModel,
    )
