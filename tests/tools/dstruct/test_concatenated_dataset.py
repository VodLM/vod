from __future__ import annotations

import numpy as np
import pytest
from vod_tools import dstruct


@pytest.fixture
def arr() -> np.ndarray:
    """A random array."""
    return np.random.rand(100, 10)


@pytest.fixture
def splits(seed: int) -> list[int]:
    """A random list of split points."""
    rgn = np.random.RandomState(seed)
    nsplits = rgn.randint(1, 10)
    return sorted(rgn.choice(range(10, 90), size=nsplits, replace=False).tolist())


@pytest.fixture
def concat_array(arr: np.ndarray, splits: list[int]) -> dstruct.ConcatenatedSizedDataset:
    """A concatenated array given the splits."""
    split_points = [0] + splits + [arr.shape[0]]
    parts = [arr[split_points[i] : split_points[i + 1]] for i in range(len(split_points) - 1)]
    return dstruct.ConcatenatedSizedDataset[np.ndarray](parts)  # type: ignore


@pytest.fixture
def arr_slice(seed: int) -> slice:
    """A random slice."""
    rgn = np.random.RandomState(seed)
    start_point = rgn.randint(0, 90)
    end_point = rgn.randint(start_point + 1, 100)
    stride = None  # rgn.randint(1, 3)
    return slice(start_point, end_point, stride)


@pytest.fixture
def items(seed: int) -> list[int]:
    """A random list of items."""
    rgn = np.random.RandomState(1 + seed)
    nitems = rgn.randint(1, 10)
    return rgn.choice(range(1, 100), size=nitems, replace=False).tolist()


@pytest.mark.parametrize("seed", range(100))
def test_concatenate_array_getitem(
    arr: np.ndarray, concat_array: dstruct.ConcatenatedSizedDataset, items: list[int,]
) -> None:
    """Test the `__getitem__` method."""
    for item in items:
        expected = arr[item]
        actual = concat_array[item]
        assert expected.shape == actual.shape
        assert np.allclose(expected, actual)

    expected = arr[items]
    actual = concat_array[items]
    assert expected.shape == actual.shape
    assert np.allclose(expected, actual)


@pytest.mark.parametrize("seed", range(100))
def test_concatenate_array_slice(
    arr: np.ndarray,
    concat_array: dstruct.ConcatenatedSizedDataset,
    arr_slice: slice,
) -> None:
    """Test the `__getitem__` method."""
    expected = arr[arr_slice]
    actual = concat_array[arr_slice]
    assert expected.shape == actual.shape
    assert np.allclose(expected, actual)
