from typing import Any, Optional

import rich
import torch
from datashape import discover, dshape


def _tensor_info(x: torch.Tensor) -> str:
    return (
        f"torch, {x.device}, "
        f"req.grads: {x.requires_grad}, "
        f"grads: {x.grad is not None}, "
        f"nans: {torch.isnan(x).sum()}, "
        f"inf: {torch.isinf(x).sum()}"
    )


@discover.register(torch.Tensor)
def _handle_tensor(m):
    shape_str = " * ".join(str(i) for i in m.shape)
    dtype_str = str(m.dtype).replace("torch.", "")
    meta = _tensor_info(m)
    ds = dshape(f"{shape_str} * {dtype_str}")
    return ds, meta


def print_pipe(
    batch: dict,
    idx: Optional[list[int]] = None,
    *,
    header: Optional[str] = None,
    header_width: int = 80,
    **_: Any,
) -> dict:
    """Print a batch of data."""
    ds = discover(batch)

    sep = header_width * "-"
    if header:
        center = header_width // 2
        header = f" {header} "
        n_left = center - len(header) // 2
        n_right = header_width - n_left - len(header)
        header_str = n_left * "=" + header + n_right * "="
        rich.print(header_str)
    rich.print(sep)
    rich.print(ds)
    rich.print(sep)
    return {}
