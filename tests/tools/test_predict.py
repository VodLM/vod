from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pytest
import tensorstore
import torch

from raffle_ds_research.tools import predict


@dataclasses.dataclass
class VectorDataset:
    x: np.ndarray
    y: np.ndarray

    def __getitem__(self, item):
        return {"x": self.x[item], "y": self.y[item]}

    def __len__(self) -> int:
        return len(self.x)

    @classmethod
    def from_config(cls, dset_size: int = 1000, vector_size: int = 32, seed: int = 1) -> "VectorDataset":
        rgn = np.random.RandomState(seed)
        y = rgn.randn(dset_size, vector_size)
        x = np.arange(dset_size)
        return cls(x=x, y=y)


@pytest.fixture
def data(dset_size: int = 1000, vector_size: int = 32, seed: int = 1) -> VectorDataset:
    return VectorDataset.from_config(dset_size=dset_size, vector_size=vector_size, seed=seed)


class Array(torch.nn.Module):
    def __init__(self, y: np.ndarray, output_key: Optional[str] = None):
        super().__init__()
        self.y = torch.nn.Parameter(torch.from_numpy(y), requires_grad=False)
        self.output_key = output_key

    def forward(self, batch: dict) -> dict[str, torch.Tensor] | torch.Tensor:
        ids = batch["x"]
        y = self.y[ids]
        if self.output_key is None:
            return y
        else:
            return {self.output_key: y}


def _collate(examples: Iterable[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    x = np.stack([item["x"] for item in examples])
    return {"x": x}


@pytest.mark.parametrize("model_output_key", [None, "y", "y.vector"])
def test_predict(tmpdir: str | Path, data: VectorDataset, model_output_key: Optional[str]):
    model = Array(y=data.y, output_key=model_output_key)

    stores = predict(
        {"train": data, "valid": data},
        cache_dir=tmpdir,
        model=model,
        model_output_key=model_output_key,
        collate_fn=_collate,
        loader_kwargs={"batch_size": 10, "num_workers": 0},
    )

    store: tensorstore.TensorStore = stores["valid"].open()

    # test that the store is correct
    y_retrieved = store[:].read().result()
    for i in range(len(data)):
        assert np.allclose(y_retrieved[i], data.y[i])
