"""A collection of Pipes and Collates: transform batches of data."""
from __future__ import annotations

from .compose import Sequential  # noqa: F401
from .debug import pprint_batch, pprint_supervised_retrieval_batch  # noqa: F401
from .hashing import _register_special_hashers, fingerprint_torch_module  # noqa: F401
from .protocols import Collate, Pipe  # noqa: F401
from .template import template_pipe  # noqa: F401
from .tokenize import tokenize_pipe, torch_tokenize_collate, torch_tokenize_pipe  # noqa: F401
from .wrappers import Partial, filter_inputs_wrapper, key_map_wrapper, retain_inputs_wrapper  # noqa: F401

# make sure to register the custom hashers, so `datasets.fingerprint.Hasher.hash` can use them.
# this is used cache the results of `datasets.Dataset.map` and co.
_register_special_hashers()
