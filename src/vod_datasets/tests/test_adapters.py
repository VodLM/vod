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


@pytest.mark.parametrize(
    "inputs,adapter_cls,output_model",
    [
        (
            [
                {
                    "id": "1",
                    "query": "What is the answer to life, the universe, and everything?",
                    "answers": ["42"],
                    "answer_scores": [1.0],
                    "retrieval_ids": [],
                    "retrieval_scores": [],
                    "subset_ids": [],
                }
            ],
            IdentityQueryAdapter,
            QueryModel,
        ),
        (
            [
                {
                    "id": "1",
                    "query": "What is the answer to life, the universe, and everything?",
                    "answers": ["42"],
                    "answer_scores": [1.0],
                    "retrieval_ids": ["1"],
                    "retrieval_scores": [1.0],
                    "subset_ids": ["xyz"],
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
                    "answer_scores": [1.0],
                    "retrieval_ids": ["1"],
                    "retrieval_scores": [1.0],
                    "subset_ids": ["xyz"],
                    "contexts": [
                        "The answer to life, the universe, and everything is 42.",
                    ],
                }
            ],
            IdentityQueryWithContextAdapter,
            QueryWithContextsModel,
        ),
        (
            [  # cais/mmlu
                {
                    "question": (
                        "You are pushing a truck along a road. "
                        "Would it be easier to accelerate this truck on Mars? Why? (Assume there is no friction)"
                    ),
                    "subject": "astronomy",
                    "choices": [
                        "It would be harder since the truck is heavier on Mars.",
                        "It would be easier since the truck is lighter on Mars.",
                        "It would be harder since the truck is lighter on Mars.",
                        "It would be the same no matter where you are.",
                    ],
                    "answer": 3,
                }
            ],
            MultipleChoiceQueryAdapter,
            QueryModel,
        ),
        (  # emozilla/quality
            [
                {
                    "article": (
                        "THE GIRL IN HIS MIND\nBy ROBERT F. YOUNG\n\n\n "
                        "[Transcriber's Note: This etext was produced from\n\n Worlds of Tomorrow April 1963\n\n "
                        "Extensive research did not uncover any evidence that\n\n the U.S."
                    ),
                    "question": (
                        "How much time has passed between Blake's night with Eldoria "
                        "and his search for Sabrina York in his mind-world?"
                    ),
                    "options": ["7 years", "10 hours", "12 years", "1 hour"],
                    "answer": 1,
                    "hard": False,
                }
            ],
            MultipleChoiceQueryWithContextAdapter,
            QueryWithContextsModel,
        ),
        (
            [  # race/middle
                {
                    "example_id": "middle2177.txt",
                    "article": (
                        "It is well-known that the prom, a formal dance held at the end of high school or "
                        "college, is an important date in every student's life. What is less well-known "
                        "is that the word prom comes from the verb to promenade, which means to walk around, "
                        "beautifully dressed, in order to attract attention. The idea is that you should see "
                        "and be seen by others.\nThe prom is not just an American tradition, "
                        "though most people believe that it started in America. In Canada the event "
                        "is called a formal. In Britain and Australia the old fashioned word dance "
                        "is more and more frequently being referred to as a prom. "
                    ),
                    "answer": "B",
                    "question": 'In which country is the prom called a "formal"?',
                    "options": ["America.", "Canada.", "Britain.", "Australia."],
                }
            ],
            MultipleChoiceQueryWithContextAdapter,
            QueryWithContextsModel,
        ),
        (
            [  # ag_news
                {
                    "text": (
                        "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, "
                        "Wall Street's dwindling\\band of ultra-cynics, are seeing green again."
                    ),
                    "label": 2,
                }
            ],
            MissingFieldQueryAdapter,
            QueryModel,
        ),
        (
            [  # ag_news
                {
                    "text": (
                        "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, "
                        "Wall Street's dwindling\\band of ultra-cynics, are seeing green again."
                    ),
                    "label": 2,
                }
            ],
            MissingFieldSectionAdapter,
            SectionModel,
        ),
        (
            [  # nq_open
                {
                    "question": "when was the last time anyone was on the moon",
                    "answer": ["14 December 1972 UTC", "December 1972"],
                }
            ],
            MissingFieldQueryAdapter,
            QueryModel,
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
            [  # squad_v2
                {
                    "id": "56ddde6b9a695914005b9628",
                    "title": "Normans",
                    "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni)",
                    "question": "In what country is Normandy located?",
                    "answers": {"text": ["France", "France", "France", "France"], "answer_start": [159, 159, 159, 159]},
                }
            ],
            SquadQueryAdapter,
            QueryModel,
        ),
        (
            [  # squad_v2
                {
                    "id": "56ddde6b9a695914005b9628",
                    "title": "Normans",
                    "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni)",
                    "question": "In what country is Normandy located?",
                    "answers": {"text": ["France", "France", "France", "France"], "answer_start": [159, 159, 159, 159]},
                }
            ],
            SquadQueryWithContextsAdapter,
            QueryWithContextsModel,
        ),
        (
            [  # Muennighoff/flan
                {
                    "inputs": 'Write an email with the subject line "EnronOnline- Change to Autohedge".',
                    "targets": "Effective Monday, October 22, 2001 the following changes will be made to the Autohedge",
                    "task": "aeslc_10templates",
                }
            ],
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
            [  # trivia_qa
                {
                    "question": "Which Lloyd Webber musical premiered in the US on 10th December 1993?",
                    "question_id": "tc_33",
                    "question_source": "http://www.triviacountry.com/",
                    "entity_pages": {
                        "doc_source": ["TagMe"],
                        "filename": ["Andrew_Lloyd_Webber.txt"],
                        "title": ["Andrew Lloyd Webber"],
                        "wiki_context": [
                            (
                                "Andrew Lloyd Webber, Baron Lloyd-Webber   (born 22 March 1948) "
                                "is an English composer and impresario of musical theatre."
                            )
                        ],
                    },
                    "search_results": {
                        "description": [],
                        "filename": [],
                        "rank": [],
                        "title": [],
                        "url": [],
                        "search_context": [],
                    },
                    "answer": {
                        "aliases": [
                            "Sunset Blvd",
                            "West Sunset Boulevard",
                            "Sunset Boulevard",
                            "Sunset Bulevard",
                            "Sunset Blvd.",
                        ],
                        "normalized_aliases": [
                            "sunset boulevard",
                            "sunset bulevard",
                            "west sunset boulevard",
                            "sunset blvd",
                        ],
                        "matched_wiki_entity_name": "",
                        "normalized_matched_wiki_entity_name": "",
                        "normalized_value": "sunset boulevard",
                        "type": "WikipediaEntity",
                        "value": "Sunset Boulevard",
                    },
                }
            ],
            TriviaQaQueryAdapter,
            QueryModel,
        ),
        (
            [  # trivia_qa
                {
                    "question": "Which Lloyd Webber musical premiered in the US on 10th December 1993?",
                    "question_id": "tc_33",
                    "question_source": "http://www.triviacountry.com/",
                    "entity_pages": {
                        "doc_source": ["TagMe"],
                        "filename": ["Andrew_Lloyd_Webber.txt"],
                        "title": ["Andrew Lloyd Webber"],
                        "wiki_context": [
                            (
                                "Andrew Lloyd Webber, Baron Lloyd-Webber   "
                                "(born 22 March 1948) is an English composer and impresario of musical theatre."
                            )
                        ],
                    },
                    "search_results": {
                        "description": [],
                        "filename": [],
                        "rank": [],
                        "title": [],
                        "url": [],
                        "search_context": [],
                    },
                    "answer": {
                        "aliases": [
                            "Sunset Blvd",
                            "West Sunset Boulevard",
                            "Sunset Boulevard",
                            "Sunset Bulevard",
                            "Sunset Blvd.",
                        ],
                        "normalized_aliases": [
                            "sunset boulevard",
                            "sunset bulevard",
                            "west sunset boulevard",
                            "sunset blvd",
                        ],
                        "matched_wiki_entity_name": "",
                        "normalized_matched_wiki_entity_name": "",
                        "normalized_value": "sunset boulevard",
                        "type": "WikipediaEntity",
                        "value": "Sunset Boulevard",
                    },
                }
            ],
            TriviaQaQueryWithContextsAdapter,
            QueryWithContextsModel,
        ),
    ],
)
def test_adapter(
    inputs: list[dict[str, typ.Any]],
    adapter_cls: Type[Adapter],
    output_model: Type[pydantic.BaseModel],
    num_proc: int = 2,
    max_rows: int = 10,
) -> None:
    """A generic test for parsing a dataset."""
    if isinstance(inputs, list):
        data = datasets.Dataset.from_list(inputs)
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
