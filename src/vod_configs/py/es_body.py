from __future__ import annotations

import copy
from typing import Any, TypeVar

from loguru import logger

ROW_IDX_KEY = "__row_idx__"
BODY_KEY: str = "__body__"
SUBSET_ID_KEY: str = "__subset_id__"
SECTION_ID_KEY: str = "__section_id__"
AUTO_STOPWORDS_KEYWORD = "__auto_stopwords__"
ES_DEFAULT_BODY = {
    "mappings": {
        "properties": {
            BODY_KEY: {
                "type": "text",
            },
            SUBSET_ID_KEY: {
                "type": "keyword",
                "ignore_above": 1_024,
            },
            SECTION_ID_KEY: {
                "type": "keyword",
                "ignore_above": 1_024,
            },
            ROW_IDX_KEY: {
                "type": "unsigned_long",
            },
        },
    },
}

LANGUAGES_MAP = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "ja": "japanese",
    "zh": "chinese",
    "ar": "arabic",
    "da": "danish",
    "nl": "dutch",
    "fi": "finnish",
    "hu": "hungarian",
    "no": "norwegian",
    "ro": "romanian",
    "sv": "swedish",
    "tr": "turkish",
    "id": "indonesian",
    "ms": "malay",
    "vi": "vietnamese",
    "th": "thai",
    "cs": "czech",
    "el": "greek",
    "is": "icelandic",
    "pl": "polish",
    "sk": "slovak",
    "sl": "slovenian",
    "et": "estonian",
    "lv": "latvian",
    "lt": "lithuanian",
}


def _normalize_language(language: str) -> str:
    """Generate the stop words configuration."""
    language = language.lower()
    if language in LANGUAGES_MAP:
        return LANGUAGES_MAP[language]
    raise NotImplementedError(f"Language `{language}` is not supported.")


T = TypeVar("T")


def _search_value_in_json_struct(x: Any, value: Any) -> bool:  # noqa: ANN401
    """Return True if `value` can be found in x."""
    if isinstance(x, dict):
        return any(_search_value_in_json_struct(y, value) for y in x.values())  # type: ignore
    if isinstance(x, list):
        return any(_search_value_in_json_struct(y, value) for y in x)  # type: ignore

    return x == value


def _replace_value_in_json_struct(x: T, old_value: Any, new_value: Any) -> T:  # noqa: ANN401
    """Replace a value in a json structure."""
    if isinstance(x, dict):
        return {key: _replace_value_in_json_struct(y, old_value, new_value) for key, y in x.items()}  # type: ignore
    if isinstance(x, list):
        return [_replace_value_in_json_struct(y, old_value, new_value) for y in x]  # type: ignore

    if x == old_value:
        return new_value

    return x


def validate_es_body(body: None | dict, language: None | str) -> dict:
    """Validate the elasticsearch body using the default values."""
    if body is None:
        body = ES_DEFAULT_BODY
    body = copy.deepcopy(body)
    if "mappings" not in body:
        body["mappings"] = ES_DEFAULT_BODY["mappings"]
        return body

    # Check the mappings
    mappings = body["mappings"]
    if "properties" not in mappings:
        mappings["properties"] = ES_DEFAULT_BODY["mappings"]["properties"]
        return body

    # Check the properties
    properties = mappings["properties"]
    for key in [BODY_KEY, SUBSET_ID_KEY, SECTION_ID_KEY, ROW_IDX_KEY]:
        if key not in properties:
            properties[key] = ES_DEFAULT_BODY["mappings"]["properties"][key]

    # Create a custom `language` stop word_filter
    if language is not None and _search_value_in_json_struct(body, AUTO_STOPWORDS_KEYWORD):
        logger.debug(
            f"Patching ES body with stop words for language `{language}` (triggered with `{AUTO_STOPWORDS_KEYWORD}`)"
        )
        language = _normalize_language(language)
        filter_name = f"custom_{language}_stop"

        # Update the filters
        settings = body.setdefault("settings", {})
        analysis = settings.setdefault("analysis", {})
        analysis_filter = analysis.setdefault("filter", {})
        if filter_name not in analysis_filter:
            analysis_filter[filter_name] = {
                "type": "stop",
                "stopwords": f"_{language}_",
                "ignore_case": "true",
            }

        # Find and replace `__auto_stop__`
        body = _replace_value_in_json_struct(
            body,
            AUTO_STOPWORDS_KEYWORD,
            filter_name,
        )

    return body
