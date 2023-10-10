"""Implementation of the command line interface."""
__version__ = "0.1.0"

from .hydra import (
    hyra_conf_path,
    init_hydra_config,
    register_omgeaconf_resolvers,
)
from .train import run_exp
