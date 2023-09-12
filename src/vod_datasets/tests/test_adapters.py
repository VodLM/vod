import dataclasses
import typing as typ

import datasets
import pydantic
import pytest
from typing_extensions import Type
from vod_datasets.rosetta.adapters import (
    Adapter,
    IdentityQueryAdapter,
    IdentityQueryWithContextAdapter,
    IdentitySectionAdapter,
    MissingFieldQueryAdapter,
    MissingFieldSectionAdapter,
    MultipleChoiceQueryAdapter,
    MultipleChoiceQueryWithContextAdapter,
    RenameQueryAdapter,
    RenameSectionAdapter,
    SquadQueryAdapter,
    SquadQueryWithContextsAdapter,
    TextToTextQueryAdapter,
    TriviaQaQueryAdapter,
    TriviaQaQueryWithContextsAdapter,
)
from vod_datasets.rosetta.models import (
    QueryModel,
    QueryWithContextsModel,
    SectionModel,
)


@dataclasses.dataclass
class Args:
    """A simple dataclass for passing arguments to the test."""

    name: str
    subset: str | None = None
    split: str | None = None

    def load(self) -> datasets.Dataset:
        """Load the dataset."""
        data = datasets.load_dataset(self.name, self.subset, split=self.split)
        if isinstance(data, datasets.DatasetDict):
            data = data[list(data.keys())[0]]

        return data  # type: ignore

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.name}.{self.subset}:{self.split}"


@pytest.mark.parametrize(
    "inputs,adapter_cls,output_model",
    [
        (
            [
                {
                    "id": "1",
                    "query": "What is the answer to life, the universe, and everything?",
                    "answers": ["42"],
                }
            ],
            IdentityQueryAdapter,
            QueryModel,
        ),
        (
            [
                {
                    "id": "1",
                    "title": "The meaning of life",
                    "content": "The answer to life, the universe, and everything is 42.",
                }
            ],
            IdentitySectionAdapter,
            SectionModel,
        ),
        (
            [
                {
                    "id": "1",
                    "query": "What is the answer to life, the universe, and everything?",
                    "titles": ["The meaning of life"],
                    "answers": ["42"],
                    "contexts": [
                        "The answer to life, the universe, and everything is 42.",
                    ],
                }
            ],
            IdentityQueryWithContextAdapter,
            QueryWithContextsModel,
        ),
        (
            Args("cais/mmlu", "astronomy", "dev"),
            MultipleChoiceQueryAdapter,
            QueryModel,
        ),
        (
            Args("emozilla/quality", None, "validation[:10]"),
            MultipleChoiceQueryWithContextAdapter,
            QueryWithContextsModel,
        ),
        (
            Args("race", "middle", "test[:10]"),
            MultipleChoiceQueryWithContextAdapter,
            QueryWithContextsModel,
        ),
        (
            Args("ag_news", None, "train[:10]"),
            MissingFieldQueryAdapter,
            QueryModel,
        ),
        (
            Args("ag_news", None, "train[:10]"),
            MissingFieldSectionAdapter,
            SectionModel,
        ),
        (
            Args("nq_open", None, "validation[:10]"),
            MissingFieldQueryAdapter,
            QueryModel,
        ),
        (
            [
                [
                    {
                        "id": "1",
                        "question": "What is the meaning of life?",
                        "answers": ["42"],
                        "answer_scores": [1.0],
                    }
                ],
                RenameQueryAdapter,
                QueryModel,
            ]
        ),
        (
            [
                {
                    "id": "1",
                    "article": "What is the meaning of life?",
                    "header": "The meaning of life",
                }
            ],
            RenameSectionAdapter,
            SectionModel,
        ),
        (
            Args("squad_v2", None, "validation[:10]"),
            SquadQueryAdapter,
            QueryModel,
        ),
        (
            Args("squad_v2", None, "validation[:10]"),
            SquadQueryWithContextsAdapter,
            QueryWithContextsModel,
        ),
        (
            Args("Muennighoff/flan", None, "test[:10]"),
            TextToTextQueryAdapter,
            QueryModel,
        ),
        (
            [
                {
                    "question": "What is the meaning of life?",
                    "answer": "42",
                }
            ],
            TextToTextQueryAdapter,
            QueryModel,
        ),
        (
            Args("trivia_qa", "rc.wikipedia", "validation[:10]"),
            TriviaQaQueryAdapter,
            QueryModel,
        ),
        (
            Args("trivia_qa", "rc.wikipedia", "validation[:10]"),
            TriviaQaQueryWithContextsAdapter,
            QueryWithContextsModel,
        ),
    ],
)
def test_adapter(
    inputs: list[dict[str, typ.Any]] | Args,
    adapter_cls: Type[Adapter],
    output_model: Type[pydantic.BaseModel],
    num_proc: int = 2,
    max_rows: int = 10,
) -> None:
    """A generic test for parsing a dataset."""
    if isinstance(inputs, list):
        data = datasets.Dataset.from_list(inputs)
    elif isinstance(inputs, Args):
        data = inputs.load()
    else:
        raise NotImplementedError(f"Cannot handle {type(inputs)}")

    # Iterate over rows
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise NotImplementedError("Only dicts are supported")

        if not adapter_cls.can_handle(row):
            row_types = {k: type(v) for k, v in row.items()}
            raise AssertionError(
                f"Cannot handle row: {row_types} using adapter `{adapter_cls}` "
                f"with input_model {adapter_cls.input_model.model_fields.keys()}"
            )

        # Test translating a row
        output = adapter_cls.translate_row(row)
        assert isinstance(output, output_model), f"Expected {output_model}, got {type(output)}"  # noqa: S101

        if i >= max_rows:
            break

    # Attempt translating the entire dataset and validate the first row
    adapted = adapter_cls.translate(data, map_kwargs={"num_proc": num_proc})
    output_model(**adapted[0])
