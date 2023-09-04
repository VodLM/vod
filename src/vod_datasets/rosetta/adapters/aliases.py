import pydantic

QUERY_ID_ALIASES = pydantic.AliasChoices(
    "id",
    "uid",
    "question_id",
    "query_id",
)
SECTION_ID_ALIASES = pydantic.AliasChoices(
    "id",
    "uid",
    "section_id",
    "context_id",
    "passage_id",
    "document.idx",
)
QUERY_ALIASES = pydantic.AliasChoices(
    "text",
    "query",
    "question",
)
SECTION_ALIASES = pydantic.AliasChoices(
    "text",
    "passage",
    "context",
    "section",
    "content",
    "article",
    "document.text",
    "body",
)
CONTEXTS_ALIASES = pydantic.AliasChoices(
    "contexts",
    "context",
    "passages",
    "passage",
    "sections",
    "section",
    "contents",
    "content",
    "articles",
    "article",
)
TITLES_ALIASES = pydantic.AliasChoices(
    "titles",
    "title",
    "heading",
    "header",
    "document.title",
)
ANSWER_ALIASES = pydantic.AliasChoices(
    "answer",
    "answers",
    "response",
)
ANSWER_SCORES_ALIASES = pydantic.AliasChoices(
    "answer_scores",
)
ANSWER_CHOICE_IDX_ALIASES = pydantic.AliasChoices(
    "answer",
    "answer_idx",
    "answer_index",
)
CHOICES_ALIASES = pydantic.AliasChoices(
    "choices",
    "options",
    "candidates",
)

INPUT_TEXTS = pydantic.AliasChoices(
    "inputs",
    "prompt",
    "question",
    "query",
)
TARGET_TEXTS = pydantic.AliasChoices(
    "targets",
    "answer",
    "response",
    "completion",
)
