import datasets
from vod_datasets.rosetta import models
from vod_datasets.tests import utils

from src.vod_datasets.rosetta.adapters.identity import (
    IdentityQueryAdapter,
    IdentityQueryWithContextAdapter,
    IdentitySectionAdapter,
)


def test_identity_dummy_queries() -> None:
    """Test parsing dummy queries."""
    dummy_queries = [
        {
            "id": "1",
            "query": "What is the meaning of life?",
            "answer": ["42"],
        },
    ]
    data = datasets.Dataset.from_list(dummy_queries)
    return utils.test_parse_dataset(
        data=data,
        adapter_cls=IdentityQueryAdapter,
        output_model=models.QueryModel,
    )


def test_identity_dummy_sections() -> None:
    """Test parsing dummy sections."""
    dummy_sections = [
        {
            "id": "1",
            "title": "title1",
            "content": "text1",
        },
    ]
    data = datasets.Dataset.from_list(dummy_sections)
    return utils.test_parse_dataset(
        data=data,
        adapter_cls=IdentitySectionAdapter,
        output_model=models.SectionModel,
    )


def test_identity_dummy_query_with_contexts() -> None:
    """Test parsing dummy query with contexts."""
    dummy_data = [
        {
            "id": "1",
            "query": "What is the meaning of life?",
            "answer": ["42"],
            "contexts": [
                "context1",
            ],
            "titles": ["title1"],
        },
    ]
    data = datasets.Dataset.from_list(dummy_data)
    return utils.test_parse_dataset(
        data=data,
        adapter_cls=IdentityQueryWithContextAdapter,
        output_model=models.QueryWithContextsModel,
    )
