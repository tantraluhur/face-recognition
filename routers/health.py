"""GET /api/health and root info endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from config import MATCH_THRESHOLD, MODEL_CATALOG
from schemas import HealthResponse, ModelInfo

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models=[
            ModelInfo(
                name=name,
                description=str(info["description"]),
                size_mb=int(info["size_mb"]),  # type: ignore[arg-type]
            )
            for name, info in MODEL_CATALOG.items()
        ],
        threshold=MATCH_THRESHOLD,
    )


@router.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "Face Recognition Backend",
        "docs": "/docs",
        "health": "/api/health",
        "verify": "POST /api/verify",
    }
