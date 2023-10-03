import os
import pathlib
import socket
import subprocess
import typing as typ
from random import randint

import randomname
import torch
from omegaconf import OmegaConf

SEED = randint(0, 100_000)  # noqa: N806, S311


def register_omgeaconf_resolvers() -> None:  # noqa: C901, PLR0915
    """Register OmegaConf resolvers. Resolvers a dynamically computed values that can be used in the config."""
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

    def _reverse_frank_split(x: str) -> str:
        if x.startswith("frank.A"):
            return x.replace("frank.A", "frank.B.")
        if x.startswith("frank.B"):
            return x.replace("frank.B", "frank.A.")
        return x

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

    # Register resolvers
    OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
    OmegaConf.register_new_resolver("hostname", socket.gethostname)
    OmegaConf.register_new_resolver("getcwd", os.getcwd)
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("int_mul", _int_mul)
    OmegaConf.register_new_resolver("int_add", _add_int)
    OmegaConf.register_new_resolver("int_div", _int_div)
    OmegaConf.register_new_resolver("int_max", _int_max)
    OmegaConf.register_new_resolver("n_gpus", lambda *_: n_gpus)
    OmegaConf.register_new_resolver("n_devices", lambda: max(1, n_gpus))
    OmegaConf.register_new_resolver("git_hash", _git_revision_hash)
    OmegaConf.register_new_resolver("git_hash_short", _git_revision_short_hash)
    OmegaConf.register_new_resolver("git_branch_name", _git_branch_name)
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("os_expanduser", os.path.expanduser)
    OmegaConf.register_new_resolver("rdn_name", randomname.get_name)
    OmegaConf.register_new_resolver("default_trainer_accelerator", _default_trainer_accelerator)
    OmegaConf.register_new_resolver("default_trainer_single_device", _default_trainer_single_device)
    OmegaConf.register_new_resolver("infer_model_type", _infer_model_type)
    OmegaConf.register_new_resolver("randint", randint)
    OmegaConf.register_new_resolver("global_seed", lambda *_: SEED)
    OmegaConf.register_new_resolver("fmt_mn", _format_model_name)
    OmegaConf.register_new_resolver("reverse_frank_split", _reverse_frank_split)
    OmegaConf.register_new_resolver("is_cuda_available", torch.cuda.is_available)
    OmegaConf.register_new_resolver("null_cls", lambda *_: None)
    OmegaConf.register_new_resolver("join_path", _join_path)
    OmegaConf.register_new_resolver("abs_path", lambda x: pathlib.Path(x).absolute().as_posix())
    OmegaConf.register_new_resolver("n_cpus", lambda *_: n_cpus)
    OmegaConf.register_new_resolver("normalize_dtype", _normalize_dtype)
