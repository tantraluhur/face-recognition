"""Face Recognition Backend entrypoint.

Run: uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import EAGER_MODELS, configure_logging
from routers import health, verify
from services.model_registry import ModelRegistry, fix_antelopev2_nesting

log = configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    log.info("Starting backend...")
    fix_antelopev2_nesting()

    registry = ModelRegistry()
    for name in EAGER_MODELS:
        await registry.preload(name)
    app.state.models = registry

    log.info("Backend ready. Docs: http://localhost:8000/docs")
    try:
        yield
    finally:
        log.info("Shutting down")
        registry.clear()


app = FastAPI(
    title="Face Recognition Backend",
    description="InsightFace-powered face comparison with buffalo_l and antelopev2",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(verify.router)
app.include_router(health.router)
