import copy
import re
import typing as typ
from typing import Any

import datasets
import jinja2

MATCH_JINJA_VARS = re.compile(r"{{\s*(\w+)\s*}}")


class Template:
    """A template for rendering multiple variables into a string.

    Lazily instantiate the jinja template to avoid pickling issues.
    """

    template: str
    _input_vars: list[str]
    _jinja_template: None | jinja2.Template = None

    def __init__(self, template: str) -> None:
        """Initialize the template."""
        self.template = template
        self._input_vars = list(MATCH_JINJA_VARS.findall(self.template))

    def render(self, row: dict[str, typ.Any]) -> str:
        """Render the template."""
        return self.jinja_template.render(**{key: row[key] for key in self.input_vars})

    def render_batch(self, batch: dict[str, typ.Any]) -> list[str]:
        """Render a batch."""
        input_vars = self.input_vars
        return [self.render({k: batch[k][i] for k in input_vars}) for i in range(len(batch[input_vars[0]]))]

    def __call__(self, inputs: typ.Iterable[typ.Any] | dict[str, list[typ.Any]], **kws: Any) -> list[str]:
        """Render a batch of inputs."""
        if isinstance(inputs, dict):
            return self.render_batch(inputs)

        return [self.render(ex) for ex in inputs]

    def __repr__(self) -> str:
        """Get the string representation."""
        return f"Template({self.template!r})"

    ## Utilities

    @property
    def input_vars(self) -> list[str]:
        """Get the template variables."""
        return copy.copy(self._input_vars)

    def is_valid(self, row: dict[str, typ.Any] | datasets.Dataset) -> bool:
        """Validate a row."""
        input_keys = set(row.keys()) if isinstance(row, dict) else set(row.features.keys())
        return all(var in input_keys for var in self.input_vars)

    ## Handle pickling of the Jinja template

    def __getstate__(self) -> dict[str, typ.Any]:
        """Get the state for pickling."""
        state = self.__dict__.copy()
        state["_jinja_template"] = None
        return state

    def __setstate__(self, state: dict[str, typ.Any]) -> None:
        """Set the state after unpickling."""
        self.__dict__.update(state)

    @property
    def jinja_template(self) -> jinja2.Template:
        """Get the jinja template."""
        if self._jinja_template is None:
            self._jinja_template = jinja2.Template(self.template)
        return self._jinja_template
