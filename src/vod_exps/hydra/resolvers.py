import os
import pathlib
import socket
import subprocess
import typing as typ
from random import randint

import omegaconf as omg
import randomname
import torch
from vod_configs import __version__ as VERSION

SEED = randint(0, 100_000)  # noqa: N806, S311


def register_omgeaconf_resolvers() -> None:  # noqa: C901, PLR0915
    """Register omg.OmegaConf resolvers. Resolvers a dynamically computed values that can be used in the config."""
    n_gpus = torch.cuda.device_count()  # noqa: N806
    n_cpus = os.cpu_count()  # noqa: N806

    def _default_trainer_accelerator(*args: typ.Any, **kwargs: typ.Any) -> str:
        if n_gpus == 0:
            return "cpu"

        return "gpu"

    def _default_trainer_single_device(*args: typ.Any, **kwargs: typ.Any) -> str:
        if n_gpus == 0:
            return "cpu"
        if n_gpus == 1:
            return "cuda:0"
        raise ValueError("N_GPUS > 1. Please specify the device.")

    def _infer_model_type(model_name: str) -> str:
        known_model_types = ["bert", "t5"]
        for model_type in known_model_types:
            if model_name.startswith(model_type):
                return model_type

        raise ValueError(
            f"Unknown mode name: {model_name}. " f"The model name should start with one of {known_model_types}."
        )

    def _format_model_name(model_name: str) -> str:
        *_, model_name = model_name.split("/")
        return model_name

    def _join_path(*args: typ.Any) -> str:
        return pathlib.Path(*args).as_posix()

    def _cmd_check_output(args: list[str]) -> str:
        """Returns the output of a command as a string."""
        try:
            output = subprocess.check_output(args, stderr=subprocess.DEVNULL)
            return output.decode("ascii").strip()
        except Exception:
            return "unknown"

    def _git_revision_hash() -> str:
        """Return the git revision hash."""
        return _cmd_check_output(["git", "rev-parse", "HEAD"])

    def _git_revision_short_hash() -> str:
        """Return the git revision hash (shortened)."""
        return _cmd_check_output(["git", "rev-parse", "--short", "HEAD"])

    def _git_branch_name() -> str:
        """Return the git branch name."""
        return _cmd_check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    def _int_div(a: int, *b: int) -> int:
        """Divide a by b. Return an integer."""
        try:
            y = a
            for x in b:
                y = y / x
        except Exception as exc:
            raise ValueError(f"Inputs: a={a}, b={b}") from exc
        return int(y)

    def _int_mul(a: int, *b: int) -> int:
        """Multiply a by b. Return an integer."""
        y = a
        for x in b:
            y *= x
        return int(y)

    def _add_int(a: int, *b: int) -> int:
        """Add a and b. Return an integer."""
        y = a
        for x in b:
            y += x
        return int(y)

    def _int_max(a: int, *b: int) -> int:
        """Return the maximum of a and b. Return an integer."""
        y = a
        for x in b:
            y = max(x, y)
        return int(y)

    def _normalize_dtype(x: str | int) -> str:
        return {
            "float16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "bf16-mixed": "bfloat16",
            "float32": "float32",
            "16": "float16",
            "32": "float32",
        }[str(x)]

    def _infer_accumulate_grad_batches(batch_size: int, per_device: int) -> int:
        """Infer the accumulate_grad_batches parameter for the Trainer assuming a single node distributed setting."""
        effective_batch_size = int(os.environ.get("WORLD_SIZE", "1")) * per_device
        if effective_batch_size > batch_size:
            return 1
        return -(-batch_size // effective_batch_size)

    def _parse_encoder_name(model_config: omg.DictConfig) -> str:
        return model_config.encoder.pretrained_model_name_or_path

    def _parse_lm_name(model_config: omg.DictConfig) -> None | str:
        if "lm" not in model_config:
            return None
        return model_config.lm.pretrained_model_name_or_path

    def _parse_model_name(model_config: omg.DictConfig) -> str:
        encoder_name = _parse_encoder_name(model_config)
        lm_name = _parse_lm_name(model_config)
        if lm_name is None:
            return f"ranker-{encoder_name}"
        return f"realm-{lm_name}-{encoder_name}"

    # Register resolvers
    omg.OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
    omg.OmegaConf.register_new_resolver("hostname", socket.gethostname)
    omg.OmegaConf.register_new_resolver("getcwd", os.getcwd)
    omg.OmegaConf.register_new_resolver("int_mul", _int_mul)
    omg.OmegaConf.register_new_resolver("int_add", _add_int)
    omg.OmegaConf.register_new_resolver("int_div", _int_div)
    omg.OmegaConf.register_new_resolver("int_max", _int_max)
    omg.OmegaConf.register_new_resolver("n_gpus", lambda *_: n_gpus)
    omg.OmegaConf.register_new_resolver("n_devices", lambda: max(1, n_gpus))
    omg.OmegaConf.register_new_resolver("git_hash", _git_revision_hash)
    omg.OmegaConf.register_new_resolver("git_hash_short", _git_revision_short_hash)
    omg.OmegaConf.register_new_resolver("git_branch_name", _git_branch_name)
    omg.OmegaConf.register_new_resolver("eval", eval)
    omg.OmegaConf.register_new_resolver("bool", bool)
    omg.OmegaConf.register_new_resolver("int", int)
    omg.OmegaConf.register_new_resolver("float", float)
    omg.OmegaConf.register_new_resolver("os_expanduser", os.path.expanduser)
    omg.OmegaConf.register_new_resolver("rdn_name", randomname.get_name)
    omg.OmegaConf.register_new_resolver("default_trainer_accelerator", _default_trainer_accelerator)
    omg.OmegaConf.register_new_resolver("default_trainer_single_device", _default_trainer_single_device)
    omg.OmegaConf.register_new_resolver("infer_model_type", _infer_model_type)
    omg.OmegaConf.register_new_resolver("randint", randint)
    omg.OmegaConf.register_new_resolver("global_seed", lambda *_: SEED)
    omg.OmegaConf.register_new_resolver("fmt_mn", _format_model_name)
    omg.OmegaConf.register_new_resolver("is_cuda_available", torch.cuda.is_available)
    omg.OmegaConf.register_new_resolver("null_cls", lambda *_: None)
    omg.OmegaConf.register_new_resolver("join_path", _join_path)
    omg.OmegaConf.register_new_resolver("abs_path", lambda x: pathlib.Path(x).absolute().as_posix())
    omg.OmegaConf.register_new_resolver("n_cpus", lambda *_: n_cpus)
    omg.OmegaConf.register_new_resolver("normalize_dtype", _normalize_dtype)
    omg.OmegaConf.register_new_resolver("infer_accumulate_grad_batches", _infer_accumulate_grad_batches)
    omg.OmegaConf.register_new_resolver("parse_encoder_name", _parse_encoder_name)
    omg.OmegaConf.register_new_resolver("parse_lm_name", _parse_lm_name)
    omg.OmegaConf.register_new_resolver("parse_model_name", _parse_model_name)
    omg.OmegaConf.register_new_resolver("code_version", lambda *_: VERSION)
