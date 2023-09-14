from datasets.fingerprint import Hasher

from .hashing import (
    fingerprint_torch_module,
    register_special_hashers,
)

# make sure to register the custom hashers, so `datasets.fingerprint` can use them.
# this is used cache the results of `datasets.Dataset.map` and co.
register_special_hashers()


def fingerprint(value: object) -> str:
    """Fingerprint a value."""
    return Hasher().hash(value)
