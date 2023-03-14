import dataclasses
from typing import Union, Any, Optional

cimport cython
import numpy as np
import torch
from cython.parallel cimport prange
from typing_extensions import TypeAlias

ctypedef long long DTYPE
NP_DTYPE: TypeAlias = np.int64
PyArray: TypeAlias = Union[np.ndarray, list, torch.Tensor]


def _cast_array(x: Union[np.ndarray, torch.Tensor, list, Any], *, dtype: np.dtype) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif isinstance(x, torch.Tensor):
        return x.numpy().astype(dtype)
    elif isinstance(x, list):
        return np.asarray(x, dtype=dtype)
    else:
        return np.asarray(x, dtype=dtype)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int _unique_by_label(
    const DTYPE[:] values,
    const DTYPE[:] labels,
    DTYPE[:, :] buffer,
    const int n_labels,
) nogil:
    """Find unique values and their labels."""

    cdef:
        unsigned int cursor = 0
        unsigned int i
        DTYPE v_i
        DTYPE l_i
        DTYPE buffered_value
        unsigned int n = len(values)
        unsigned int found = 0

    for i in range(n):
        v_i = values[i]
        if v_i < 0:
            # ignore negative values (padding)
            continue
        l_i = labels[i]
        found = 0
        for s in range(cursor):
            buffered_value = buffer[s, 0]
            if buffered_value == v_i:
                buffer[s, 1 + l_i] += 1
                found = 1
                break

        if found == 0:
            buffer[cursor, 0] = v_i
            buffer[cursor, 1:] = 0
            buffer[cursor, 1 + l_i] += 1
            cursor += 1

    return cursor


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef unsigned int [:] _batched_unique_by_label(
    const DTYPE[:, :] values,
    const DTYPE[:, :] labels,
    DTYPE[:, :, :] buffer,
    unsigned int[:] cursors,
    const int batch_size,
    const int n_labels,
) nogil:

    cdef:
        int i

    for i in prange(batch_size, nogil=True):
        cursors[i] = _unique_by_label(
            values[i],
            labels[i],
            buffer[i],
            n_labels,
        )

    return cursors

@dataclasses.dataclass
class Frequencies:
    """Label frequencies for a set of unique values."""
    values: np.ndarray
    counts: np.ndarray

    __annotations__= {
            "values": np.ndarray,
            "counts": np.ndarray,
        }

    def __getitem__(self, item):
        return Frequencies(values=self.values[item], counts=self.counts[item])


    def __iter__(self):
        return zip(self.values, self.counts)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"Frequencies(values={self.values}, counts={self.counts})"


def get_frequencies(
    values: PyArray,
    *,
    labels: Optional[PyArray] = None,
    n_labels: Optional[int] = None,
    max_n_unique: Optional[int] = None,
) -> Frequencies:


    # create arrays
    values = _cast_array(values, dtype=NP_DTYPE)
    if labels is not None:
        if n_labels is None:
            raise ValueError("`n_labels` must be provided if `labels` are provided.")
        labels = _cast_array(labels, dtype=NP_DTYPE)
    else:
        labels = np.zeros_like(values, dtype=NP_DTYPE)

    # check shapes
    if labels.max() >= n_labels:
        raise ValueError(f"labels.max() ({labels.max()}) >= n_labels ({n_labels})")
    if values.shape != labels.shape:
        raise ValueError(f"values.shape ({values.shape}) != labels.shape ({labels.shape})")

    # check dimensions
    if max_n_unique is None:
        max_n_unique = values.shape[-1]
    ndim = len(values.shape)
    if ndim == 1:
        batched = False
    elif ndim == 2:
        batched = True
    else:
        raise ValueError(f"values dimension must be 1 or 2. Found shape {values.shape}.")


    if not batched:
        buffer = np.full((max_n_unique, 1 + n_labels), dtype=np.int64, fill_value=-1)
        cursor = _unique_by_label(
            values,
            labels,
            buffer,
            n_labels,
        )
        uvalues =  np.asarray(buffer)
        return Frequencies(values=uvalues[:cursor, 0], counts=uvalues[:cursor, 1:])

    else:
        batch_size = len(values)
        buffer = np.full((batch_size, max_n_unique, 1 + n_labels), dtype=NP_DTYPE, fill_value=-1)
        cursors = np.zeros(batch_size, dtype=np.uint32)

        # run the C function
        cursors = _batched_unique_by_label(
            values,
            labels,
            buffer,
            cursors,
            batch_size,
            n_labels,
        )
        uvalues = np.asarray(buffer)
        cursors = np.asarray(cursors)
        max_cursor = cursors.max()
        return Frequencies(values=uvalues[:, :max_cursor, 0],
                           counts=uvalues[:, :max_cursor, 1:])
