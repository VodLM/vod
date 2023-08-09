from __future__ import annotations

import numbers
import os
import warnings
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
from lightning.pytorch.loggers import WandbLogger
from vod_tools import interfaces
from vod_tools.misc.config import config_to_flat_dict

from src import vod_models

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
    transformers.utils.logging.set_verbosity_error()  # type: ignore
    torch.set_float32_matmul_precision("medium")

    try:
        torch._dynamo.config.log_level = os.environ.get("DYNAMO_LOG_LEVEL", "INFO")  # type: ignore
    except AttributeError:
        warnings.warn("Could not set torch._dynamo.config.log_level", stacklevel=2)

    # torch.multiprocessing.set_sharing_strategy("file_system")


def _get_ranker_meta_data(ranker: vod_models.Ranker) -> dict[str, Any]:
    return {
        "n_trainable_params": sum(p.numel() for p in ranker.parameters() if p.requires_grad),
        "n_total_params": sum(p.numel() for p in ranker.parameters()),
        "output_shape_question": list(ranker.encoder.get_output_shape(interfaces.FieldType.QUERY)),
        "output_shape_section": list(ranker.encoder.get_output_shape(interfaces.FieldType.SECTION)),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),  # type: ignore
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),  # type: ignore
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),  # type: ignore
    }


def log_config(
    config: dict[str, Any] | omegaconf.DictConfig,
    exp_dir: Path,
    extras: dict[str, Any],
    fabric: L.Fabric,
) -> None:
    """Log the config as hyperparameters and save it locally."""
    config_path = Path(exp_dir, "config.yaml")
    all_data: dict[str, Any] = omegaconf.OmegaConf.to_container(config, resolve=True)  # type: ignore
    all_data.update(extras)
    with config_path.open("w") as f:
        yaml.dump(all_data, f)

    for logger in fabric.loggers:
        if isinstance(logger, WandbLogger):
            flat_config = config_to_flat_dict(all_data, sep="/")
            flat_config = {k: _cast_hps(v) for k, v in flat_config.items()}
            logger.experiment.config.update(flat_config)


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
