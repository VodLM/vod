import typing as typ

import torch


def repr_tensor(x: torch.Tensor) -> str:
    """Return a string representation of a tensor."""
    return f"Tensor(shape={x.shape}, dtype={x.dtype}, device={x.device})"


def human_format_bytes(x: int, unit: typ.Literal["KB", "MB", "GB"]) -> str:
    """Format bytes to a human readable format."""
    divider = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }
    return f"{x / divider[unit]:.2f} {unit}"


def human_format_nb(num: int | float, precision: int = 2, base: float = 1000.0) -> str:
    """Converts a number to a human-readable format."""
    magnitude = 0
    while abs(num) >= base:
        magnitude += 1
        num /= base
    # add more suffixes if you need them
    q = ["", "K", "M", "G", "T", "P"][magnitude]
    return f"{num:.{precision}f}{q}"
