# pylint: disable=no-member

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Union

import pydantic
import tensorstore
from pydantic import BaseModel


class TensorStoreKvStoreConfig(BaseModel):
    """Configuration for a TensorStore key-value store."""

    class Config:
        """pydantic config for TensorStoreKvStoreConfig"""

        allow_mutation = False
        extra = pydantic.Extra.forbid

    driver: Literal["file"]
    path: str

    @pydantic.validator("path", pre=True)
    def _validate_path(cls, value: Any) -> str:
        return str(Path(value).expanduser().absolute())


class TensorStoreFactory(BaseModel):
    """This class represents a TensorStore configuration.
    Open a store using the `open` method."""

    class Config:
        """pydantic config for TensorStoreFactory"""

        allow_mutation = False
        extra = pydantic.Extra.forbid

    driver: Literal["n5", "zarr"]
    kvstore: TensorStoreKvStoreConfig
    metadata: dict[str, Any]

    def open(self, create: int = False, delete_existing: int = False, **kwargs: Any) -> tensorstore.TensorStore:
        """Open and return a TensorStore."""
        cfg = self.dict()
        future = tensorstore.open(cfg, create=create, delete_existing=delete_existing, **kwargs)
        return future.result()

    @classmethod
    def from_factory(
        cls,
        path: Union[str, Path],
        shape: Iterable[int],
        chunk_size: int = 100,
        driver: Literal["n5", "zarr"] = "zarr",
        dtype: Literal["float16", "float32", "float64"] = "float32",
    ) -> "TensorStoreFactory":
        """Instantiate a `TensorStoreFactory` from a path and shape."""
        shape = tuple(shape)
        driver_meta = {
            "n5": {
                "dataType": dtype,
                "dimensions": shape,
                "compression": {"type": "gzip"},
                "blockSize": [chunk_size, *shape[1:]],
            },
            "zarr": {
                "dtype": {"float16": "<f2", "float32": "<f4", "float64": "<f8"}[dtype],
                "shape": shape,
                "chunks": [chunk_size, *shape[1:]],
            },
        }
        metadata = driver_meta[driver]
        instance = cls(
            driver=driver,
            kvstore=TensorStoreKvStoreConfig(driver="file", path=path),
            metadata=metadata,
        )
        return instance
