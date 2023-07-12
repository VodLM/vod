from __future__ import annotations

import dataclasses
import functools
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional, Type, TypeVar

import lightning as L
import numpy as np
import pytest
import tensorstore
import torch

from src.vod_tools import predict

T = TypeVar("T")


@dataclasses.dataclass
class VectorDataset:
    """A dataset of random vectors `x` with labels `y`."""

    x: np.ndarray
    y: np.ndarray

    def __getitem__(self, item: int | slice | Iterable[int]) -> dict[str, Any]:
        """Return a row of data."""
        return {"x": self.x[item], "y": self.y[item]}

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.x)

    @classmethod
    def from_config(cls: Type[T], dset_size: int = 1000, vector_size: int = 32, seed: int = 1) -> T:
        """Create a dataset from a configuration."""
        rgn = np.random.RandomState(seed)
        y = rgn.randn(dset_size, vector_size)
        x = np.arange(dset_size)
        return cls(x=x, y=y)


@pytest.fixture
def data(dset_size: int = 1000, vector_size: int = 32, seed: int = 1) -> VectorDataset:
    """Create a dataset."""
    return VectorDataset.from_config(dset_size=dset_size, vector_size=vector_size, seed=seed)


class Array(torch.nn.Module):
    """A simple model that returns a vector from an array."""

    def __init__(self, y: np.ndarray, output_key: Optional[str] = None):
        super().__init__()
        self.y = torch.nn.Parameter(torch.from_numpy(y), requires_grad=False)
        self.output_key = output_key

    def forward(self, batch: dict) -> dict[str, torch.Tensor] | torch.Tensor:
        """Forward pass."""
        ids = batch["x"]
        y = self.y[ids]
        if self.output_key is None:
            return y

        return {self.output_key: y}


def _collate(examples: Iterable[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    x = np.stack([item["x"] for item in examples])
    return {"x": x}


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("model_output_key", [None, "y", "y.vector"])
def test_predict(tmpdir: str | Path, data: VectorDataset, model_output_key: Optional[str]) -> None:
    """Test that the prediction works as expected."""
    model = Array(y=data.y, output_key=model_output_key)
    predict_fn = functools.partial(
        predict.predict,
        data,
        fabric=L.Fabric(),
        cache_dir=tmpdir,
        model=model,
        model_output_key=model_output_key,
        collate_fn=_collate,
        loader_kwargs={"batch_size": 10, "num_workers": 0},
    )

    stores = {}
    for key in ["try", "first", "second"]:
        if key == "try":
            with pytest.raises(FileNotFoundError):
                stores[key] = predict_fn(
                    open_mode="r",  # try to read: expect to fail
                )
            continue
        stores[key] = predict_fn(
            open_mode="x" if key == "first" else "r",  # write at first pass, then only read
        )
        if not stores[key].exists():
            raise FileNotFoundError(f"Store {key} was not created.")

    store: tensorstore.TensorStore = stores["first"].open()

    # test that the store is correct
    y_retrieved = store[:].read().result()
    for i in range(len(data)):
        assert np.allclose(y_retrieved[i], data.y[i])  # noqa: S101


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_predict(
            str(tmpdir),
            data=VectorDataset.from_config(dset_size=1_000, vector_size=32, seed=1),
            model_output_key=None,
        )
