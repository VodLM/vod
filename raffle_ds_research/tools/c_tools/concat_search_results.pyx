import dataclasses
from typing import Union, Optional

cimport cython
import numpy as np
import torch
from cython.parallel cimport prange
from typing_extensions import TypeAlias

ctypedef long long DTYPE_LONG
ctypedef float DTYPE_FLOAT
NP_DTYPE_LONG: TypeAlias = np.int64
NP_DTYPE_FLOAT: TypeAlias = np.float32
PyArray: TypeAlias = Union[np.ndarray, list, torch.Tensor]


@dataclasses.dataclass
class ConcatenatedTopk:
    """Label frequencies for a set of unique values."""

    indices: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    features: np.ndarray

    __annotations__ = {
        "indices": np.ndarray,
        "scores": np.ndarray,
        "labels": np.ndarray,
        "features": np.ndarray,
    }

    def __getitem__(self, item):
        return ConcatenatedTopk(
            indices=self.indices[item],
            scores=self.scores[item],
            labels=self.labels[item],
            features=self.features[item],
        )

    def __iter__(self):
        for idx, sco, lbl, fea in zip(self.indices, self.scores, self.labels, self.features):
            yield type(self)(indices=idx, scores=sco, labels=lbl, features=fea)

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"{type(self).__name__}(indices={self.indices}, scores={self.scores}, labels={self.labels})"


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_LONG _concat_single(
    DTYPE_LONG [:] ab_indices,
    DTYPE_FLOAT [:] ab_scores,
    DTYPE_LONG [:] ab_labels,
    DTYPE_FLOAT [:] ab_features,
    DTYPE_LONG [:] a_indices,
    DTYPE_FLOAT [:] a_scores,
    DTYPE_FLOAT [:] a_features,
    DTYPE_LONG [:] b_indices,
    DTYPE_FLOAT [:] b_scores,
    DTYPE_FLOAT [:] b_features,
    unsigned int total,
    unsigned int max_a,
) nogil:

    cdef:
        unsigned long i
        DTYPE_LONG j = 0
        DTYPE_LONG buffered_idx
        DTYPE_FLOAT buffered_score

    for i in range(len(a_indices)):
        buffered_idx = a_indices[i]
        if buffered_idx < 0 or j >= total or j >= max_a:
            break
        ab_indices[j] = buffered_idx
        ab_scores[j] = a_scores[i]
        ab_features[j] = a_features[i]
        ab_labels[j] = 0
        j += 1

    for i in range(len(b_indices)):
        buffered_idx = b_indices[i]
        if buffered_idx < 0 or j >= total:
            break
        ab_indices[j] = buffered_idx
        ab_scores[j] = b_scores[i]
        ab_features[j] = b_features[i]
        ab_labels[j] = 1
        j += 1

    return j


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_LONG [:]  _concat_batch(
    DTYPE_LONG [:, :] ab_indices,
    DTYPE_FLOAT [:, :] ab_scores,
    DTYPE_LONG [:, :] ab_labels,
    DTYPE_FLOAT [:, :] ab_features,
    DTYPE_LONG [:, :] a_indices,
    DTYPE_FLOAT [:, :] a_scores,
    DTYPE_FLOAT [:, :] a_features,
    DTYPE_LONG [:, :] b_indices,
    DTYPE_FLOAT [:, :] b_scores,
    DTYPE_FLOAT [:, :] b_features,
    DTYPE_LONG [:] cursors,
    unsigned long total,
    unsigned long max_a,
) nogil:
    cdef unsigned long i
    cdef unsigned long batch_size = a_indices.shape[0]
    for i in prange(batch_size, nogil=True):
        cursors[i] = _concat_single(
            ab_indices[i],
            ab_scores[i],
            ab_labels[i],
            ab_features[i],
            a_indices[i],
            a_scores[i],
            a_features[i],
            b_indices[i],
            b_scores[i],
            b_features[i],
            total,
            max_a,
        )

    return cursors


def concat_unique_topk(
    a_indices: np.ndarray,
    a_scores: np.ndarray,
    a_features: np.ndarray,
    b_indices: np.ndarray,
    b_scores: np.ndarray,
    b_features: np.ndarray,
    total: int,
    max_a: Optional[int] = None,
    truncate: bool = True,
) -> ConcatenatedTopk:

    # check input
    if max_a is None:
        max_a = len(a_indices)
    else:
        max_a = min(max_a, len(a_indices))
    assert a_indices.shape == a_scores.shape
    assert b_indices.shape == b_scores.shape

    # Sort A
    a_sorted_ids = np.argsort(a_scores, axis=-1)
    a_sorted_ids = np.flip(a_sorted_ids, axis=-1)[..., :max_a]
    a_indices = np.take_along_axis(a_indices, a_sorted_ids, axis=-1)
    a_scores = np.take_along_axis(a_scores, a_sorted_ids, axis=-1)
    a_features = np.take_along_axis(a_features, a_sorted_ids, axis=-1)

    # Sort B
    b_sorted_ids = np.argsort(b_scores, axis=-1)
    b_sorted_ids = np.flip(b_sorted_ids, axis=-1)
    b_indices = np.take_along_axis(b_indices, b_sorted_ids, axis=-1)
    b_scores = np.take_along_axis(b_scores, b_sorted_ids, axis=-1)
    b_features = np.take_along_axis(b_features, b_sorted_ids, axis=-1)


    # create the buffer and run the cython function
    if len(a_indices.shape) == 1:
        ab_indices = np.full((total,), dtype=NP_DTYPE_LONG, fill_value=-1)
        ab_scores = np.full((total,), dtype=NP_DTYPE_FLOAT, fill_value=np.nan)
        ab_labels = np.full((total,), dtype=NP_DTYPE_LONG, fill_value=-1)
        ab_features = np.full((total,), dtype=NP_DTYPE_FLOAT, fill_value=np.nan)
        cursor = _concat_single(
            ab_indices,
            ab_scores,
            ab_labels,
            ab_features,
            a_indices.astype(NP_DTYPE_LONG),
            a_scores.astype(NP_DTYPE_FLOAT),
            a_features.astype(NP_DTYPE_FLOAT),
            b_indices.astype(NP_DTYPE_LONG),
            b_scores.astype(NP_DTYPE_FLOAT),
            b_features.astype(NP_DTYPE_FLOAT),
            total,
            max_a,
        )
        if not truncate:
            cursor = None

        return ConcatenatedTopk(
            indices=np.asarray(ab_indices)[:cursor],
            scores=np.asarray(ab_scores)[:cursor],
            labels=np.asarray(ab_labels)[:cursor],
            features=np.asarray(ab_features)[:cursor],
        )

    elif len(a_indices == 2):
        batch_size = a_indices.shape[0]
        ab_indices = np.full((batch_size, total), dtype=NP_DTYPE_LONG, fill_value=-1)
        ab_scores = np.full((batch_size, total), dtype=NP_DTYPE_FLOAT, fill_value=np.nan)
        ab_labels = np.full((batch_size, total), dtype=NP_DTYPE_LONG, fill_value=-1)
        ab_features = np.full((batch_size, total), dtype=NP_DTYPE_FLOAT, fill_value=np.nan)
        cursors = np.full((batch_size,), dtype=NP_DTYPE_LONG, fill_value=-1)
        cursors = _concat_batch(
            ab_indices,
            ab_scores,
            ab_labels,
            ab_features,
            a_indices.astype(NP_DTYPE_LONG),
            a_scores.astype(NP_DTYPE_FLOAT),
            a_features.astype(NP_DTYPE_FLOAT),
            b_indices.astype(NP_DTYPE_LONG),
            b_scores.astype(NP_DTYPE_FLOAT),
            b_features.astype(NP_DTYPE_FLOAT),
            cursors,
            total,
            max_a,
        )
        if truncate:
            cursor = np.min(cursors)
        else:
            cursor = None
        return ConcatenatedTopk(
            indices=np.asarray(ab_indices)[:, :cursor],
            scores=np.asarray(ab_scores)[:, :cursor],
            labels=np.asarray(ab_labels)[:, :cursor],
            features=np.asarray(ab_features)[:, :cursor],
        )
    else:
        raise ValueError(f"a_indices must be 1D or 2D. Found shape: {a_indices.shape}")
