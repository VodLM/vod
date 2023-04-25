from __future__ import annotations

from typing import Optional

import pydantic

from raffle_ds_research.tools.index_tools.retrieval_data_type import RetrievalDataType


class FaissInitConfig(pydantic.BaseModel):
    """Configuration used to init/build a faiss index."""

    class Config:
        """pydantic config."""

        allow_mutation = False
        extra = "forbid"

    index_path: str
    nprobe: int = 8


class InitResponse(pydantic.BaseModel):
    """Response to the initialization request."""

    class Config:
        """pydantic config."""

        allow_mutation = False
        extra = "forbid"

    success: bool
    exception: Optional[str]


class SearchFaissQuery(pydantic.BaseModel):
    """Query to search a faiss index."""

    class Config:
        """pydantic config."""

        allow_mutation = False
        extra = "forbid"

    vectors: list = pydantic.Field(..., description="A batch of vectors. Implicitly defines `list[list[float]]`.")
    top_k: int = 3


class FastSearchFaissQuery(pydantic.BaseModel):
    """This is the same as SearchFaissQuery, but with the vectors serialized."""

    class Config:
        """pydantic config."""

        allow_mutation = False
        extra = "forbid"

    vectors: str = pydantic.Field(
        ..., description="A batch of serialized vectors`. Implicitly defines `np.ndarray` or `torch.Tensor`."
    )
    array_type: RetrievalDataType = RetrievalDataType.NUMPY
    top_k: int = 3


class FaissSearchResponse(pydantic.BaseModel):
    """Response to the search request."""

    class Config:
        """pydantic config."""

        allow_mutation = False
        extra = "forbid"

    scores: list = pydantic.Field(..., description="A batch of scores. Implicitly defines `list[list[float]]`.")
    indices: list = pydantic.Field(..., description="A batch of indices. Implicitly defines `list[list[int]]`.")


class FastFaissSearchResponse(pydantic.BaseModel):
    """This is the same as FaissSearchResponse, but with the vectors serialized."""

    class Config:
        """pydantic config."""

        allow_mutation = False
        extra = "forbid"

    scores: str
    indices: str
