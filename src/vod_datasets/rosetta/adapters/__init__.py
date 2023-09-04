from typing_extensions import Type
from vod_datasets.rosetta.models import DatasetType

from .base import Adapter
from .identity import (
    IdentityQueryAdapter,
    IdentityQueryWithContextAdapter,
    IdentitySectionAdapter,
)
from .mcqa import (
    MultipleChoiceQueryAdapter,
    MultipleChoiceQueryWithContextAdapter,
)
from .missing_fields import (
    MissingFieldQueryAdapter,
    MissingFieldSectionAdapter,
)
from .rename_fields import (
    RenameQueryAdapter,
    RenameSectionAdapter,
)
from .squad import (
    SquadQueryAdapter,
    SquadQueryWithContextsAdapter,
)
from .text_to_text import (
    TextToTextQueryAdapter,
)
from .trivia_qa import (
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
    MissingFieldQueryAdapter,
    MultipleChoiceQueryAdapter,
    TriviaQaQueryAdapter,
    SquadQueryAdapter,
    TextToTextQueryAdapter,
]

KNOWN_SECTION_ADAPTERS: list[Type[Adapter]] = [
    IdentitySectionAdapter,
    RenameSectionAdapter,
    MissingFieldSectionAdapter,
]

ROSETTA_ADAPTERS: dict[DatasetType, list[Type[Adapter]]] = {
    "queries_with_context": KNOWN_QUERY_WITH_CONTEXT_ADAPTERS,
    "queries": KNOWN_QUERY_ADAPTERS,
    "sections": KNOWN_SECTION_ADAPTERS,
}
