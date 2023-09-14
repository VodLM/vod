import typing as typ

import vod_types as vt

PREDICT_IDX_COL_NAME: str = "__idx__"

T = typ.TypeVar("T")
Ti = typ.TypeVar("Ti", bound=typ.Any | int)


class WithIndices(vt.DictsSequence[Ti]):
    """This class is used to add the column `PREDICT_IDX_COL_NAME` to the batch."""

    def __init__(self, dataset: vt.DictsSequence[Ti]):
        self.dataset = dataset

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, Ti]:
        """Returns a row from the dataset."""
        batch: dict[str, Ti] = self.dataset[item]
        if PREDICT_IDX_COL_NAME in batch:
            raise ValueError(
                f"Column `{PREDICT_IDX_COL_NAME}` already exists in batch (keys={batch.keys()}. "
                f"Cannot safely add the row index."
            )

        batch[PREDICT_IDX_COL_NAME] = item  # type: ignore
        return batch

    def __iter__(self) -> typ.Iterable[dict]:
        """Returns an iterator over the rows of the dataset."""
        for i in range(len(self)):
            yield self[i]


def _safely_fetch_key(row: dict) -> int:
    try:
        return row.pop(PREDICT_IDX_COL_NAME)
    except KeyError as exc:
        raise ValueError(
            f"Column {PREDICT_IDX_COL_NAME} not found in batch. "
            f"Make sure to wrap your dataset with `DatasetWithIndices`."
        ) from exc


def _collate_with_indices(
    examples: typ.Iterable[dict[str, typ.Any]], *, collate_fn: vt.Collate, **kws: typ.Any
) -> dict[str, typ.Any]:
    ids = [_safely_fetch_key(row) for row in examples]
    batch = collate_fn(examples, **kws)  # type: ignore
    batch[PREDICT_IDX_COL_NAME] = ids
    return batch


class CollateWithIndices(vt.Collate):
    """Wraps a `Collate` to add the column `IDX_COL` to the batch."""

    def __init__(self, collate_fn: vt.Collate):  # type: ignore
        self.collate_fn = collate_fn

    def __call__(self, examples: typ.Iterable[dict[str, typ.Any]], **kws: typ.Any) -> dict[str, typ.Any]:
        """Collate the rows along with the row indixes (`IDX_COL`)."""
        return _collate_with_indices(examples, collate_fn=self.collate_fn, **kws)  # type: ignore
