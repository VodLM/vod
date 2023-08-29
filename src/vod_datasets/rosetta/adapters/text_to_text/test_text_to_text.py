import datasets
from vod_datasets.rosetta import models, testing

from .adapter import TextToTextQueryAdapter


def test_flan_as_queries() -> None:
    """Test parsing the FLAN dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset("Muennighoff/flan", split="test[:10]"),  # type: ignore
        adapter_cls=TextToTextQueryAdapter,
        output_model=models.QueryModel,
    )  # type: ignore


def test_dummy_as_queries() -> None:
    """Test parsing a dummy dataset."""
    data = datasets.Dataset.from_list(
        [
            {
                "question": "What is the meaning of life?",
                "answer": "42",
            },
        ]
    )
    return testing.test_parse_dataset(
        data,  # type: ignore
        adapter_cls=TextToTextQueryAdapter,
        output_model=models.QueryModel,
    )  # type: ignore
