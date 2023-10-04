import typing as typ

import lightning as L
import torch
import vod_models


class OnFirstBatchCallback(typ.Protocol):
    """A callback that is called on the first batch of the first epoch."""

    def __call__(self, fabric: L.Fabric, batch: dict[str, torch.Tensor], ranker: vod_models.Ranker) -> None:
        """Do some stuff."""
        ...
