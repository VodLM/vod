from __future__ import annotations

from pydantic import BaseModel


class CustomRequest(BaseModel):
    """Request example."""

    content: str


class CustomResponse(BaseModel):
    """Response example."""

    content: str
