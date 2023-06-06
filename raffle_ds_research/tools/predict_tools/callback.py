from __future__ import annotations

from asyncio import Future
from typing import Any, List, Optional

import lightning.pytorch as pl
import tensorstore
import tensorstore as ts
import torch

from .wrappers import PREDICT_IDX_COL_NAME


class StorePredictions:
    """Context manager to store the predictions of a model to a `ts.TensorStore`."""

    def __init__(
        self,
        trainer: pl.Trainer,
        store: tensorstore.TensorStore,
        model_output_key: Optional[str] = None,
    ):
        self.trainer = trainer
        self.callback = TensorStoreCallback(store, model_output_key)

    def __enter__(self) -> None:
        """Add the callback to the trainer."""
        self.trainer.callbacks.append(self.callback)  # type: ignore

    def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> None:  # noqa: ANN401
        """Remove the callback from the trainer."""
        self.trainer.callbacks.remove(self.callback)  # type: ignore


class TensorStoreCallback(pl.Callback):
    """Allows storing the output of each `prediction_step` into a `ts.TensorStore`."""

    def __init__(
        self,
        store: ts.TensorStore,
        output_key: Optional[str] = None,
    ):
        self.store = store
        self.output_key = output_key

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,  # noqa: ARG002
        outputs: dict[str, Any],
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Store the outputs of the prediction step to the cache."""
        try:
            indices = batch[PREDICT_IDX_COL_NAME]
        except KeyError as exc:
            raise KeyError(
                f"Column {PREDICT_IDX_COL_NAME} not found in batch. "
                f"Make sure to wrap your dataset with `DatasetWithIndices`."
            ) from exc
        vectors = _select_vector_from_output(outputs, self.output_key)
        _write_vectors_to_store(
            self.store,
            vectors=vectors,
            idx=indices,
            asynchronous=False,
        )


def _write_vectors_to_store(
    store: ts.TensorStore,
    vectors: torch.Tensor,
    idx: List[int],
    asynchronous: bool = False,
) -> Optional[Future]:
    """Write vectors to a `TensorStore`."""
    if idx is None:
        raise ValueError("idx must be provided")
    vectors = vectors.detach().cpu().numpy()
    dtype = store.spec().dtype
    vectors = vectors.astype(dtype.numpy_dtype)

    if asynchronous:
        return store[idx].write(vectors)

    store[idx] = vectors
    return None


def _select_vector_from_output(batch: torch.Tensor | dict, key: Optional[str] = None) -> torch.Tensor:
    if key is None:
        if isinstance(batch, dict):
            raise TypeError("Input batch is a dictionary, the argument `field` must be set.")
        return batch

    if key not in batch:
        raise ValueError(f"Key {key} not found in batch. Found {batch.keys()}")

    return batch[key]
