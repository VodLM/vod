from __future__ import annotations

import numbers
import os
from pathlib import Path
from typing import Any, TypeVar

import datasets
import faiss
import lightning as L
import loguru
import omegaconf
import torch
import transformers
import yaml
from lightning.fabric.loggers.logger import Logger as FabricLogger

from raffle_ds_research.core.ml import Ranker
from raffle_ds_research.utils.config import config_to_flat_dict

T = TypeVar("T")


def _do_nothing(*args: Any, **kwargs: Any) -> None:
    """Do nothing."""


def _identity(value: T) -> T:
    """Return the same value."""
    return value


def _cast_hps(value: object) -> str:
    """Cast a value to a string."""
    formatter = {
        numbers.Number: _identity,
        str: _identity,
        Path: lambda x: str(x.absolute()),
    }.get(type(value), str)
    return formatter(value)


def set_training_context() -> None:
    """Set the general context for torch, datasets, etc."""
    omp_num_threads = os.environ.get("OMP_NUM_THREADS", None)
    if omp_num_threads is not None:
        loguru.logger.warning(f"OMP_NUM_THREADS={omp_num_threads}")
        torch.set_num_threads(int(omp_num_threads))
        faiss.omp_set_num_threads(int(omp_num_threads))

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    torch.set_float32_matmul_precision("medium")

    # torch.multiprocessing.set_sharing_strategy("file_system")


def _get_ranker_meta_data(ranker: Ranker) -> dict[str, Any]:
    return {
        "n_trainable_params": sum(p.numel() for p in ranker.parameters() if p.requires_grad),
        "n_total_params": sum(p.numel() for p in ranker.parameters()),
        "output_shape": ranker.encoder.get_output_shape(),  # type: ignore
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
    }


def log_config(config: dict[str, Any] | omegaconf.DictConfig, exp_dir: Path, extras: dict[str, Any]) -> None:
    """Log the config as hyperparameters and save it locally."""
    config_path = Path(exp_dir, "config.yaml")
    all_data: dict[str, Any] = omegaconf.OmegaConf.to_container(config, resolve=True)  # type: ignore
    all_data.update(extras)
    with config_path.open("w") as f:
        yaml.dump(all_data, f)

    # log he config to wandb
    try:
        # pylint: disable=import-outside-toplevel
        import wandb

        try:
            flat_config = config_to_flat_dict(all_data, sep="/")
            flat_config = {k: _cast_hps(v) for k, v in flat_config.items()}
            wandb.config.update(flat_config)
        except wandb.errors.Error:
            loguru.logger.debug("Could not log config to wandb")
    except ImportError:
        ...


def init_fabric(
    *args,  # noqa: ANN002
    loggers: omegaconf.DictConfig | list[FabricLogger] | dict[str, FabricLogger],
    **kwargs,  # noqa: ANN003
) -> L.Fabric:
    """Initialize a fabric with the given `omegaconf`-defined loggers."""
    if isinstance(loggers, omegaconf.DictConfig):
        loggers = omegaconf.OmegaConf.to_container(loggers, resolve=True)  # type: ignore

    if isinstance(loggers, dict):
        loggers = list(loggers.values())

    return L.Fabric(*args, loggers=loggers, **kwargs)  # type: ignore
