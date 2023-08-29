from typing_extensions import Type
from vod_datasets.rosetta.models import DatasetType

from .adapter import Adapter
from .identity.adapter import (
    IdentityQueryAdapter,
    IdentityQueryWithContextAdapter,
    IdentitySectionAdapter,
)
from .mcqa.adapter import (
    MultipleChoiceQueryAdapter,
    MultipleChoiceQueryWithContextAdapter,
)
from .rename.adapter import (
    RenameQueryAdapter,
    RenameSectionAdapter,
)
from .squad.adapter import (
    SquadQueryAdapter,
    SquadQueryWithContextsAdapter,
)
from .text_to_text.adapter import (
    TextToTextQueryAdapter,
)
from .trivia_qa.adapter import (
    TriviaQaQueryAdapter,
    TriviaQaQueryWithContextsAdapter,
)

KNOWN_QUERY_WITH_CONTEXT_ADAPTERS: list[Type[Adapter]] = [
    IdentityQueryWithContextAdapter,
    MultipleChoiceQueryWithContextAdapter,
    TriviaQaQueryWithContextsAdapter,
    SquadQueryWithContextsAdapter,
]

KNOWN_QUERY_ADAPTERS: list[Type[Adapter]] = [
    IdentityQueryAdapter,
    RenameQueryAdapter,
    MultipleChoiceQueryAdapter,
    TriviaQaQueryAdapter,
    SquadQueryAdapter,
    TextToTextQueryAdapter,
]

KNOWN_SECTION_ADAPTERS: list[Type[Adapter]] = [
    IdentitySectionAdapter,
    RenameSectionAdapter,
]

ROSETTA_ADAPTERS: dict[DatasetType, list[Type[Adapter]]] = {
    "queries_with_context": KNOWN_QUERY_WITH_CONTEXT_ADAPTERS,
    "queries": KNOWN_QUERY_ADAPTERS,
    "sections": KNOWN_SECTION_ADAPTERS,
}
