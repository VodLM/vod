from __future__ import annotations

import typing
from typing import Optional

from vod_configs.py.utils import StrictModel


class KeyMap(StrictModel):
    """Defines the name of the keys used on the query side and on the section side."""

    query: str
    section: str


class DataLoaderConfig(StrictModel):
    """Base configuration for a pytorch DataLoader."""

    batch_size: int
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False


class BaseCollateConfig(StrictModel):
    """Defines a base configuration for the collate function."""

    query_max_length: int = 512
    section_max_length: int = 512


class RetrievalCollateConfig(BaseCollateConfig):
    """Defines a configuration for the retrieval collate function."""

    # base config
    prefetch_n_sections: int = 100
    n_sections: Optional[int] = 10
    max_pos_sections: Optional[int] = 3
    support_size: Optional[int] = None
    post_filter: Optional[str] = None
    do_sample: bool = False
    in_batch_negatives: bool = False
    in_batch_neg_offset: int = 0
    prep_num_proc: int = 4

    # name of the keys to use on the query side and on the section side
    text_keys: KeyMap = KeyMap(query="text", section="text")  #  text field
    vector_keys: KeyMap = KeyMap(query="vector", section="vector")  #  vector field
    section_id_keys: KeyMap = KeyMap(query="section_ids", section="id")  #  label field (section ids)
    group_id_keys: KeyMap = KeyMap(query="group_hash", section="group_hash")  #  group hash (kb_id, lang, etc.)


class SamplerFactoryConfig(StrictModel):
    """Configuration for a dataloader sampler."""

    mode: typing.Literal["lookup", "inverse_frequency"]
    key: str
    lookup: Optional[dict[str, float]] = None
    default_weight: float = 1.0
