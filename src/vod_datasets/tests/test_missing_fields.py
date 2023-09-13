import datasets
from vod_datasets.rosetta import models
from vod_datasets.tests import utils

from src.vod_datasets.rosetta.adapters.missing_fields import MissingFieldQueryAdapter, MissingFieldSectionAdapter


def test_agnews_as_queries() -> None:
    """Test parsing the AG News dataset."""
    return utils.test_parse_dataset(
        datasets.load_dataset(
            "ag_news",
            split="train[:10]",
        ),  # type: ignore
        adapter_cls=MissingFieldQueryAdapter,
        output_model=models.QueryModel,
    )


def test_agnews_as_sections() -> None:
    """Test parsing the AG News dataset."""
    return utils.test_parse_dataset(
        datasets.load_dataset(
            "ag_news",
            split="train[:10]",
        ),  # type: ignore
        adapter_cls=MissingFieldSectionAdapter,
        output_model=models.SectionModel,
    )


def test_nq_open_as_queries() -> None:
    """Test parsing the Natural Questions Open dataset."""
    return utils.test_parse_dataset(
        datasets.load_dataset("nq_open", split="validation[:10]"),  # type: ignore
        adapter_cls=MissingFieldQueryAdapter,
        output_model=models.QueryModel,
    )
