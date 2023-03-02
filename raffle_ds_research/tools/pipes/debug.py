from typing import Optional

import rich
import torch
from datashape import discover, dshape

from .pipe import Pipe


def torch_meta(x: torch.Tensor):
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
    meta = torch_meta(m)
    ds = dshape(f"{shape_str} * {dtype_str}")
    return ds, meta


class Print(Pipe):
    header: Optional[str] = None

    def _process_batch(self, batch: dict, idx: Optional[list[int]] = None, **kwargs) -> dict:
        ds = discover(batch)
        if self.header:
            rich.print(f"=== [bold]{self.header}[/bold] ===")
        rich.print(ds)
        return {}
