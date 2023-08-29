import datasets
from vod_datasets.rosetta import models, testing

from .adapter import MultipleChoiceQueryAdapter, MultipleChoiceQueryWithContextAdapter


def test_mmu_as_queries() -> None:
    """Test parsing the SQuaD dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset("cais/mmlu", "astronomy", split="dev"),  # type: ignore
        adapter_cls=MultipleChoiceQueryAdapter,
        output_model=models.QueryModel,
    )  # type: ignore


def test_quality_as_queries_with_context() -> None:
    """Test parsing the QuALITY dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset("emozilla/quality", split="validation[:10]"),  # type: ignore
        adapter_cls=MultipleChoiceQueryWithContextAdapter,
        output_model=models.QueryWithContextsModel,
    )


def test_race_as_queries_with_context() -> None:
    """Test parsing the RACE dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset("race", "middle", split="test[:10]"),  # type: ignore
        adapter_cls=MultipleChoiceQueryWithContextAdapter,
        output_model=models.QueryWithContextsModel,
    )
