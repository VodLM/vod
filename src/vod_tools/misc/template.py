import re
import typing

import datasets
import jinja2

MATCH_JINJA_VARS = re.compile(r"{{\s*(\w+)\s*}}")


class Template:
    """A template for rendering multiple variables into a string."""

    template: str

    def __init__(self, template: str) -> None:
        """Initialize the template."""
        self.template = template

    @property
    def input_vars(self) -> list[str]:
        """Get the template variables."""
        return list(MATCH_JINJA_VARS.findall(self.template))

    def is_valide(self, row: dict[str, typing.Any] | datasets.Dataset) -> bool:
        """Validate a row."""
        input_keys = set(row.keys()) if isinstance(row, dict) else set(row.features.keys())
        return all(var in input_keys for var in self.input_vars)

    def render(self, row: dict[str, typing.Any]) -> str:
        """Render the template."""
        return jinja2.Template(self.template).render(**{key: row[key] for key in self.input_vars})

    def __repr__(self) -> str:
        """Get the string representation."""
        return f"Template({self.template!r})"
