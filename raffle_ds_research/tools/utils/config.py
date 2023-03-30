import os
import socket
from random import randint
from typing import Optional

import randomname
import torch
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

from raffle_ds_research.tools.utils.git import git_branch_name, git_revision_hash, git_revision_short_hash
from raffle_ds_research.tools.utils.misc import int_div, int_max, int_mul


def init_hydra_config(
    config_path="rank_t5.configs",
    overrides: Optional[list[str]] = None,
    config_name: str = "main",
    return_hydra_config: bool = False,
    version_base: str = "1.3",
) -> DictConfig:
    """Initializes Hydra and loads configuration."""
    with initialize_config_module(
        config_module=config_path,
        version_base=version_base,
    ):
        if overrides is None:
            overrides = []

        config = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=return_hydra_config,
        )

    return config


def register_omgeaconf_resolvers():
    N_GPUS = torch.cuda.device_count()
    GIT_HASH = git_revision_hash()
    GIT_HASH_SHORT = git_revision_short_hash()
    GIT_BRANCH_NAME = git_branch_name()
    SEED = randint(0, 100_000)

    def _default_trainer_accelerator(*args, **kwargs):
        if N_GPUS == 0:
            return "cpu"
        else:
            return "gpu"

    def _default_trainer_single_device(*args, **kwargs):
        if N_GPUS == 0:
            return "cpu"
        elif N_GPUS == 1:
            return "cuda:0"
        else:
            raise ValueError("N_GPUS > 1. Please specify the device.")

    def _infer_model_type(model_name: str):
        known_model_types = ["bert", "t5"]
        for model_type in known_model_types:
            if model_name.startswith(model_type):
                return model_type

        raise ValueError(
            f"Unknown mode name: {model_name}. " f"The model name should start with one of {known_model_types}."
        )

    def _format_model_name(model_name: str):
        *_, model_name = model_name.split("/")
        return model_name

    # Register resolvers
    OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
    OmegaConf.register_new_resolver("hostname", socket.gethostname)
    OmegaConf.register_new_resolver("getcwd", os.getcwd)
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("int_mul", int_mul)
    OmegaConf.register_new_resolver("int_div", int_div)
    OmegaConf.register_new_resolver("int_max", int_max)
    OmegaConf.register_new_resolver("n_gpus", lambda *_: N_GPUS)
    OmegaConf.register_new_resolver("n_devices", lambda: max(1, N_GPUS))
    OmegaConf.register_new_resolver("git_hash", lambda *_: GIT_HASH)
    OmegaConf.register_new_resolver("git_hash_short", lambda *_: GIT_HASH_SHORT)
    OmegaConf.register_new_resolver("git_branch_name", lambda *_: GIT_BRANCH_NAME)
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("os_expanduser", os.path.expanduser)
    OmegaConf.register_new_resolver("rdn_name", randomname.get_name)
    OmegaConf.register_new_resolver("default_trainer_accelerator", _default_trainer_accelerator)
    OmegaConf.register_new_resolver("default_trainer_single_device", _default_trainer_single_device)
    OmegaConf.register_new_resolver("infer_model_type", _infer_model_type)
    OmegaConf.register_new_resolver("randint", randint)
    OmegaConf.register_new_resolver("global_seed", lambda *_: SEED)
    OmegaConf.register_new_resolver("fmt_mn", _format_model_name)
