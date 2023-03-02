import json
from functools import partial
from typing import Iterable, Optional

from datasets.fingerprint import hashregister
from jinja2 import Template as JinjaTemplate
from pydantic import Field

from .pipe import Pipe


def unpack_batch(batch: dict[str, list], keys: list[str]) -> Iterable[dict]:
    subset = {key: batch[key] for key in keys}
    master_key, *other_keys = subset
    for i in range(len(batch[master_key])):
        example = {key: batch[key][i] for key in keys}
        yield example


class TemplatePipe(Pipe):
    template: str = Field(
        ...,
        description="The templates to use for the completion.",
    )
    input_keys: list[str] = Field(
        ...,
        description="The keys to use for the input.",
    )
    output_key: str = Field(
        "text",
        description="The key to use for the output.",
    )

    def _process_batch(self, batch: dict, idx: Optional[list[int]] = None, **kwargs) -> dict:
        template = JinjaTemplate(self.template)
        apply_fn = partial(TemplatePipe._apply_to_example, template=template)
        unpacked_batch = unpack_batch(batch, self.input_keys)
        output = map(apply_fn, unpacked_batch)
        return {self.output_key: list(output)}

    @staticmethod
    def _apply_to_example(example: dict, *, template: JinjaTemplate) -> str:
        output = template.render(example)
        return output


@hashregister(TemplatePipe)
def _hash_template(hasher, value: TemplatePipe):
    data = value.dict()
    h = hasher.hash_bytes(json.dumps(data, sort_keys=True).encode("utf-8"))
    return h
