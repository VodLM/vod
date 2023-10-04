"""Implementation of the command line interface."""
from .hydra import (
    hyra_conf_path,
    init_hydra_config,
    register_omgeaconf_resolvers,
)
from .train import run_exp
