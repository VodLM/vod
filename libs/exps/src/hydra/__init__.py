import pathlib

from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from .resolvers import register_omgeaconf_resolvers


def hyra_conf_path() -> str:
    """Return the path to the hydra config directory."""
    return pathlib.Path(__file__).parent.as_posix()


def init_hydra_config(
    config_path: str = "exps.hydra",
    overrides: None | list[str] = None,
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

        return compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=return_hydra_config,
        )
