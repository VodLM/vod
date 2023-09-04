import datasets
from vod_datasets.rosetta import models, testing

from src.vod_datasets.rosetta.adapters.trivia_qa import TriviaQaQueryAdapter, TriviaQaQueryWithContextsAdapter


def test_trivia_qa_as_queries() -> None:
    """Test parsing the TriviaQA dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset("trivia_qa", "rc.wikipedia", split="validation"),  # type: ignore
        adapter_cls=TriviaQaQueryAdapter,
        output_model=models.QueryModel,
    )


def test_trivia_qa_as_queries_with_contexts() -> None:
    """Test parsing the TriviaQA dataset."""
    return testing.test_parse_dataset(
        datasets.load_dataset("trivia_qa", "rc.wikipedia", split="validation"),  # type: ignore
        adapter_cls=TriviaQaQueryWithContextsAdapter,
        output_model=models.QueryWithContextsModel,
    )


if __name__ == "__main__":
    test_trivia_qa_as_queries()
