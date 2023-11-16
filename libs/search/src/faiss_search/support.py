import math
import os
import re
import typing as typ
from multiprocessing import pool as mp_pool

import pydantic
from typing_extensions import Self, Type

T = typ.TypeVar("T")


def infer_factory_centroids(factory_str: str, n_vecs: int, min_vecs_per_centroid: int = 128) -> str:
    """Infer the number of centroids for a factory string containing IVFauto."""
    if "IVFauto" not in factory_str:
        return factory_str
    num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    n_centroids = max(num_threads, 2 ** math.ceil(math.log2(n_vecs / min_vecs_per_centroid)))
    return factory_str.replace("IVFauto", f"IVF{n_centroids}")


def rate_limited_imap(func: typ.Callable[[int], T], seq: typ.Iterable, workers: int = 1) -> typ.Iterable[T]:
    """A threaded imap that does not produce elements faster than they are consumed."""
    pool = mp_pool.ThreadPool(workers)
    res = None
    for i in seq:
        res_next = pool.apply_async(func, (i,))
        if res:
            yield res.get()
        res = res_next
    if res is not None:
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

    preproc: None | str = None
    n_centroids: int
    ncodes: None | int = None
    nbits: None | int = None
    encoding: typ.Literal["flat", "", "fs", "fs"] = "flat"

    @pydantic.field_validator("encoding", mode="before")
    @classmethod
    def _validate_encoding(cls: Type[Self], v: None | str) -> str:
        if v is None:
            return "flat"
        return v

    @pydantic.field_validator("preproc", mode="before")
    @classmethod
    def _validate_preproc(cls: Type[Self], v: None | str) -> None | str:
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
