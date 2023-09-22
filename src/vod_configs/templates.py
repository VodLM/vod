import re

import pydantic

from .utils import StrictModel


class TemplatesConfig(StrictModel):
    """Prompt templates."""

    query: str = pydantic.Field(
        default=r"query: {{ query }}",
        description="Template for formatting a query",
    )
    answer: str = pydantic.Field(
        default=r"answer: {{ answer }}",
        description="Template formatting answers before encoding for retrieval.",
    )
    section: str = pydantic.Field(
        default=r"passage: {{ content }}",
        description="Template formatting documents before encoding for retrieval.",
    )

    @property
    def input_variables(cls) -> set[str]:
        """Return the input variables."""
        variables = set()
        for attribute_value in cls.__dict__.values():
            matches = re.findall(r"{{\s*(.*?)\s*}}", attribute_value)
            variables.update(matches)
        return variables
