from __future__ import annotations

from typing import Any, Iterable

import lightning.pytorch as pl
import torch

from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.dataset_builder.builder import DatasetProtocol

PREDICT_IDX_COL_NAME = "__idx__"


class ModuleWrapper(pl.LightningModule):
    """This class is used to wrap a `torch.nn.Module` in a `pl.LightningModule`."""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, batch: dict) -> torch.Tensor:
        return self.module(batch)


class DatasetWithIndices(DatasetProtocol):
    """This class is used to add the column `IDX_COL` to the batch"""

    def __init__(self, dataset: DatasetProtocol):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict:
        batch = self.dataset[item]
        if PREDICT_IDX_COL_NAME in batch:
            raise ValueError(
                f"Column {PREDICT_IDX_COL_NAME} already exists in batch (keys={batch.keys()}. "
                f"Cannot safely add the row index."
            )

        batch[PREDICT_IDX_COL_NAME] = item
        return batch


def _safely_fetch_key(row: dict) -> int:
    try:
        return row.pop(PREDICT_IDX_COL_NAME)
    except KeyError:
        raise ValueError(
            f"Column {PREDICT_IDX_COL_NAME} not found in batch. "
            f"Make sure to wrap your dataset with `DatasetWithIndices`."
        )


def _collate_with_indices(examples: Iterable[dict[str, Any]], *, collate_fn: pipes.Collate, **kwargs: Any) -> dict:
    ids = [_safely_fetch_key(row) for row in examples]
    batch = collate_fn(examples, **kwargs)
    batch[PREDICT_IDX_COL_NAME] = ids
    return batch


class CollateWithIndices(pipes.Collate):
    def __init__(self, collate_fn: pipes.Collate):  # type: ignore
        self.collate_fn = collate_fn

    def __call__(self, examples: Iterable[dict[str, Any]], **kwargs: Any) -> dict:
        return _collate_with_indices(examples, collate_fn=self.collate_fn, **kwargs)


def _warp_as_lightning_model(
    model: torch.nn.Module | pl.LightningModule,
) -> pl.LightningModule:
    """Wrap the model to return IDX_COL along the batch values"""
    if isinstance(model, pl.LightningModule):
        return model
    elif isinstance(model, torch.nn.Module):
        return ModuleWrapper(model)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
