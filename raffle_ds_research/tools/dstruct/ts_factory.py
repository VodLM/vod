# pylint: disable=no-member

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Any, Iterable, Literal, Union

import datasets
import pydantic
import tensorstore as ts
from pydantic import BaseModel
from typing_extensions import Self, Type


class TensorStoreKvStoreConfig(BaseModel):
    """Configuration for a TensorStore key-value store."""

    class Config:
        """pydantic config for TensorStoreKvStoreConfig."""

        allow_mutation = False
        extra = pydantic.Extra.forbid

    driver: Literal["file"]
    path: str

    @pydantic.validator("path", pre=True)
    def _validate_path(cls, value: str | Path) -> str:
        return str(Path(value).expanduser().absolute())


class TensorStoreFactory(BaseModel):
    """This class represents a TensorStore configuration. Open a store using the `open` method."""

    class Config:
        """pydantic config for TensorStoreFactory."""

        allow_mutation = False
        extra = pydantic.Extra.forbid

    driver: Literal["n5", "zarr"]
    kvstore: TensorStoreKvStoreConfig
    metadata: dict[str, Any]

    def open(self, create: int = False, delete_existing: int = False, **kwargs: Any) -> ts.TensorStore:
        """Open and return a TensorStore."""
        cfg = self.dict()
        future = ts.open(cfg, create=create, delete_existing=delete_existing, **kwargs)
        return future.result()

    @classmethod
    def instantiate(
        cls: Type[Self],
        path: Union[str, Path],
        shape: Iterable[int],
        chunk_size: int = 100,
        driver: Literal["n5", "zarr"] = "zarr",
        dtype: Literal["float16", "float32", "float64"] = "float32",
    ) -> Self:
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
        store_factory = cls(
            driver=driver,
            kvstore=TensorStoreKvStoreConfig(driver="file", path=str(path)),
            metadata=metadata,
        )

        cfg_path = _factory_cfg_path(path)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w") as f:
            f.write(store_factory.json(indent=2))

        return store_factory

    @classmethod
    def from_path(cls: Type[Self], path: Union[str, Path]) -> Self:
        """Instantiate a `TensorStoreFactory` from a path."""
        cfg_path = _factory_cfg_path(path)
        with open(cfg_path, "r") as f:
            return cls.parse_raw(f.read())


def _factory_cfg_path(path: pathlib.Path | str) -> pathlib.Path:
    return pathlib.Path(path) / "factory.json"


@datasets.fingerprint.hashregister(TensorStoreFactory)
def _hash_store_factory_lazy_array(hasher: datasets.fingerprint.Hasher, obj: TensorStoreFactory) -> str:
    return hasher.hash(obj.json())
