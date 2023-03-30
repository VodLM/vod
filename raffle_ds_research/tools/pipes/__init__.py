from .compose import Sequential
from .debug import pprint_batch, pprint_supervised_retrieval_batch
from .hashing import _register_special_hashers, fingerprint_torch_module
from .protocols import Collate, Pipe
from .template import template_pipe
from .tokenize import tokenize_pipe, torch_tokenize_collate, torch_tokenize_pipe
from .wrappers import filter_inputs_wrapper, key_map_wrapper, retain_inputs_wrapper

# make sure to register the custom hashers, so `datasets.fingerprint.Hasher.hash` can use them.
# this is used cache the results of `datasets.Dataset.map` and co.
_register_special_hashers()
