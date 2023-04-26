from __future__ import annotations

from typing import Optional

import pydantic


class DataLoaderConfig(pydantic.BaseModel):
    """Base configuration for a pytorch DataLoader."""

    class Config:
        """pydantic config."""

        extra = pydantic.Extra.forbid

    batch_size: int
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
