from __future__ import annotations

import math
import re
from multiprocessing import pool as mp_pool
from typing import Callable, Iterable, Literal, Optional, TypeVar

import pydantic
from typing_extensions import Self, Type

T = TypeVar("T")


def infer_factory_centroids(factory_str: str, n_vecs: int, min_vecs_per_centroid: int = 64) -> str:
    """Infer the number of centroids for a factory string containing IVFauto."""
    if "IVFauto" not in factory_str:
        return factory_str
    n_centroids = max(1, 2 ** math.ceil(math.log2(n_vecs / min_vecs_per_centroid)))
    return factory_str.replace("IVFauto", f"IVF{n_centroids}")


def rate_limited_imap(func: Callable[[int], T], seq: Iterable, workers: int = 1) -> Iterable[T]:
    """A threaded imap that does not produce elements faster than they are consumed."""
    pool = mp_pool.ThreadPool(workers)
    res = None
    for i in seq:
        res_next = pool.apply_async(func, (i,))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


index_factory_pattern = re.compile(
    r"(?P<preproc>(OPQ[0-9]+(_[0-9]+)?,|PCAR[0-9]+,))?"
    r"IVF(?P<n_centroids>([0-9]+)),"
    r"(Flat|PQ(?P<ncodes>([0-9]+))"
    r"x(?P<nbits>([0-9]+))"
    r"(?P<encoding>(|fs|fsr))?)"
)


class IVFPQFactory(pydantic.BaseModel):
    """Parse an IVFPQFactory string."""

    preproc: Optional[str] = None
    n_centroids: int
    ncodes: Optional[int] = None
    nbits: Optional[int] = None
    encoding: Literal["flat", "", "fs", "fs"] = "flat"

    @pydantic.validator("encoding", pre=True)
    def _validate_encoding(cls, v):  # noqa: ANN
        if v is None:
            return "flat"
        return v

    @pydantic.validator("preproc", pre=True)
    def _validate_preproc(cls, v):  # noqa: ANN
        if v is None:
            return None
        return v.rstrip(",")

    @classmethod
    def parse(cls: Type[Self], factory: str) -> Self:
        """Parse a factory string into a IVFPQFactory object."""
        matchobject = index_factory_pattern.match(factory)

        if matchobject is None:
            raise ValueError(
                f"Invalid factory string: `{factory}`. Only IVF-PQ indices are supported (e.g., `IVF4096,PQ64x8`)"
            )

        return cls(**matchobject.groupdict())  # type: ignore

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            f"FaissFactory("
            f"Preproc={self.preproc}, "
            f"IVF=({self.n_centroids}), "
            f"PQ({self.ncodes}x{self.nbits}), encoding={self.encoding})"
        )
