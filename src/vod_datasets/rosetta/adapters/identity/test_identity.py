import datasets
from vod_datasets.rosetta import models, testing

from .adapter import IdentityQueryAdapter, IdentityQueryWithContextAdapter, IdentitySectionAdapter


def test_identity_dummy_queries() -> None:
    """Test parsing dummy queries."""
    dummy_queries = [
        {
            "uid": "1",
            "query": "What is the meaning of life?",
            "answer": ["42"],
        },
    ]
    data = datasets.Dataset.from_list(dummy_queries)
    return testing.test_parse_dataset(
        data=data,
        adapter_cls=IdentityQueryAdapter,
        output_model=models.QueryModel,
    )


def test_identity_dummy_sections() -> None:
    """Test parsing dummy sections."""
    dummy_sections = [
        {
            "uid": "1",
            "title": "title1",
            "content": "text1",
        },
    ]
    data = datasets.Dataset.from_list(dummy_sections)
    return testing.test_parse_dataset(
        data=data,
        adapter_cls=IdentitySectionAdapter,
        output_model=models.SectionModel,
    )


def test_identity_dummy_query_with_contexts() -> None:
    """Test parsing dummy query with contexts."""
    dummy_data = [
        {
            "uid": "1",
            "query": "What is the meaning of life?",
            "answer": ["42"],
            "contexts": [
                "context1",
            ],
            "titles": ["title1"],
        },
    ]
    data = datasets.Dataset.from_list(dummy_data)
    return testing.test_parse_dataset(
        data=data,
        adapter_cls=IdentityQueryWithContextAdapter,
        output_model=models.QueryWithContextsModel,
    )
