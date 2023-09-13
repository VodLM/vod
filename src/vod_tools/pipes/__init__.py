"""A collection of Pipes and Collates: transform batches of data."""


from datasets.fingerprint import Hasher

from .compose import (
    Sequential,
)
from .debug import (
    pprint_batch,
    pprint_retrieval_batch,
)
from .hashing import (
    _register_special_hashers,
    fingerprint_torch_module,
)
from .wrappers import (
    Partial,
    filter_inputs_wrapper,
    key_map_wrapper,
    retain_inputs_wrapper,
)

# make sure to register the custom hashers, so `pipes.fingerprint` can use them.
# this is used cache the results of `datasets.Dataset.map` and co.
_register_special_hashers()


def fingerprint(value: object) -> str:
    """Fingerprint a value."""
    return Hasher().hash(value)
