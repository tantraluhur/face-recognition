"""DTOs shared between routers and services."""

from __future__ import annotations

from pydantic import BaseModel

from config import MATCH_THRESHOLD


class VerifyRequest(BaseModel):
    profile_image: str  # base64, optionally with a data-URI prefix
    live_image: str


class ModelResult(BaseModel):
    similarity: float | None = None
    verified: bool | None = None
    threshold: float = MATCH_THRESHOLD
    took_ms: int = 0
    error: str | None = None


class VerifyResponse(BaseModel):
    buffalo_l: ModelResult
    antelopev2: ModelResult


class ModelInfo(BaseModel):
    name: str
    description: str
    size_mb: int


class HealthResponse(BaseModel):
    status: str
    models: list[ModelInfo]
    threshold: float
