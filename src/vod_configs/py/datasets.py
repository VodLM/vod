from __future__ import annotations

import copy
from typing import Any, Optional, Union

import omegaconf
import pydantic
import transformers
from vod_tools.misc.config import as_pyobj_validator

_DEFAULT_SPLITS = ["train", "validation"]

QUESTION_TEMPLATE = "Question: {{ text }}"
SECTION_TEMPLATE = "{% if title %}Title: {{ title }}. Document: {% endif %}{{ content }}"
DEFAULT_TEMPLATES = {
    "question": QUESTION_TEMPLATE,
    "section": SECTION_TEMPLATE,
}

_DEFAULT_SPLITS = ["train", "validation"]


FRANK_A_KBIDS = [
    4,
    9,
    10,
    14,
    20,
    26,
    29,
    30,
    32,
    76,
    96,
    105,
    109,
    130,
    156,
    173,
    188,
    195,
    230,
    242,
    294,
    331,
    332,
    541,
    598,
    1061,
    1130,
    1148,
    1242,
    1264,
    1486,
    1599,
    1663,
    1665,
]  # noqa: E501
FRANK_B_KBIDS = [
    2,
    6,
    7,
    11,
    12,
    15,
    24,
    25,
    33,
    35,
    37,
    80,
    81,
    121,
    148,
    194,
    198,
    269,
    294,
    334,
    425,
    554,
    596,
    723,
    790,
    792,
    1284,
    1584,
    1589,
]  # noqa: E501


class NamedDset(pydantic.BaseModel):
    """A dataset name with splits."""

    class Config:
        """Pydantic configuration."""

        extra = pydantic.Extra.forbid

    name: str
    split: str

    @property
    def split_alias(self) -> str:
        """Return a slightluy more human-readable version of the split name."""
        aliases = {
            "validation": "val",
        }
        return aliases.get(self.split, self.split)

    @pydantic.validator("split", pre=True)
    def _validate_split(cls, v: str) -> str:
        dictionary = {
            "train": "train",
            "val": "validation",
            "validation": "validation",
            "test": "test",
        }
        if v not in dictionary:
            raise ValueError(f"Invalid split name: {v}")
        return dictionary[v]

    def __hash__(self) -> int:
        """Hash the object based on its name and split."""
        return hash((self.name, self.split))


def parse_named_dsets(names: str | list[str], default_splits: Optional[list[str]] = None) -> list[NamedDset]:
    """Parse a string of dataset names.

    Names are `+` separated and splits are specified with `:` and separated by `-`.
    """
    if default_splits is None:
        default_splits = copy.copy(_DEFAULT_SPLITS)

    if not isinstance(names, (list, omegaconf.ListConfig)):
        names = [names]

    outputs = []
    for part in (p for parts in names for p in parts.split("+")):
        if ":" in part:
            name, splits = part.split(":")
            splits = splits.split("-")
        else:
            name = part
            splits = default_splits
        for split in splits:
            # TEMPORARY HACK
            if "frank.A" in name:
                frank_split = "A"
            elif "frank.B" in name:
                frank_split = "B"
            else:
                frank_split = None

            if frank_split is not None and name.endswith("-kb*"):
                kbids = {"A": FRANK_A_KBIDS, "B": FRANK_B_KBIDS}[frank_split]
                for kbid in kbids:
                    kb_name = name.replace("-kb*", f"-kb{kbid}")
                    outputs.append(NamedDset(name=kb_name, split=split))
            else:
                outputs.append(NamedDset(name=name, split=split))
    return outputs


class BaseDatasetFactoryConfig(pydantic.BaseModel):
    """Defines a base configuration for a retrieval dataset builder."""

    class Config:
        """Pydantic config for the `DatasetFactoryConfig` class."""

        extra = pydantic.Extra.forbid
        arbitrary_types_allowed = True

    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]
    templates: dict[str, str] = DEFAULT_TEMPLATES
    prep_map_kwargs: dict[str, Any] = {}
    subset_size: Optional[int] = None
    filter_unused_sections: bool = False
    min_section_tokens: Optional[int] = None
    group_hash_key: str = "group_hash"
    group_keys: list[str] = ["kb_id", "language"]

    # validators
    _validate_templates = pydantic.validator("templates", allow_reuse=True, pre=True)(as_pyobj_validator)
    _validate_prep_map_kwargs = pydantic.validator("prep_map_kwargs", allow_reuse=True, pre=True)(as_pyobj_validator)


class DatasetFactoryConfig(NamedDset, BaseDatasetFactoryConfig):
    """Defines a configuration for a retrieval dataset builder."""

    ...
