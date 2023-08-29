import datasets
from vod_datasets.rosetta import models, testing

from .adapter import FlanQueryAdapter


def test_squad_as_queries() -> None:
    """Test parsing the SQuaD dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset("Muennighoff/flan", split="test[:10]"),  # type: ignore
        adapter_cls=FlanQueryAdapter,
        output_model=models.QueryModel,
    )  # type: ignore
