import datasets
from vod_datasets.rosetta import models, testing

from src.vod_datasets.rosetta.adapters.rename_fields import RenameQueryAdapter, RenameSectionAdapter


def test_rename_dummy_queries() -> None:
    """Test parsing dummy queries."""
    dummy_queries = [
        {
            "id": "1",
            "question": "What is the meaning of life?",
            "answers": ["42"],
            "answer_scores": [1.0],
        },
    ]
    data = datasets.Dataset.from_list(dummy_queries)
    return testing.test_parse_dataset(
        data=data,
        adapter_cls=RenameQueryAdapter,
        output_model=models.QueryModel,
    )


def test_rename_dummy_sections() -> None:
    """Test parsing dummy sections."""
    dummy_sections = [
        {
            "id": "1",
            "article": "What is the meaning of life?",
            "header": "The meaning of life",
        },
    ]
    data = datasets.Dataset.from_list(dummy_sections)
    return testing.test_parse_dataset(
        data=data,
        adapter_cls=RenameSectionAdapter,
        output_model=models.SectionModel,
    )
