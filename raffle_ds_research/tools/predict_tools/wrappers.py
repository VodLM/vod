from __future__ import annotations

import functools
from typing import Callable, Sized

import pytorch_lightning as pl
import torch
from typing_extensions import TypeAlias

from raffle_ds_research.datasets.builders.builder import DatasetProtocol

PREDICT_IDX_COL_NAME = "__idx__"
CollateFnType: TypeAlias = Callable[[list[dict]], dict]


class ModuleWrapper(pl.LightningModule):
    """This class is used to wrap a `torch.nn.Module` in a `pl.LightningModule`."""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, batch: dict) -> torch.Tensor:
        return self.module(batch)


class DatasetWithIndices(DatasetProtocol[dict]):
    """This class is used to add the column `IDX_COL` to the batch"""

    def __init__(self, dataset: Sized):
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


def collate_with_ids(rows: list[dict], collate_fn: CollateFnType) -> dict:
    def _safely_fetch_id(row: dict) -> int:
        try:
            return row.pop(PREDICT_IDX_COL_NAME)
        except KeyError:
            raise ValueError(
                f"Column {PREDICT_IDX_COL_NAME} not found in batch. "
                f"Make sure to wrap your dataset with `DatasetWithIndices`."
            )

    ids = [_safely_fetch_id(row) for row in rows]
    batch = collate_fn(rows)
    batch[PREDICT_IDX_COL_NAME] = ids
    return batch


def _wrap_collate_fn_with_indices(collate_fn: CollateFnType) -> CollateFnType:
    """Wrap the collate_fn to return IDX_COL along the batch values"""

    return functools.partial(collate_with_ids, collate_fn=collate_fn)


def _wrap_dataset_with_indices(dataset: Sized) -> DatasetWithIndices:
    """Wrap the dataset to return IDX_COL along the batch values"""
    return DatasetWithIndices(dataset)


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
