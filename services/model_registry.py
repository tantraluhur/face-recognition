"""Loads and manages InsightFace models."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

import insightface

from config import CTX_CPU, DET_SIZE

log = logging.getLogger(__name__)

FaceModel = insightface.app.FaceAnalysis


class ModelRegistry:
    """Holds loaded InsightFace models.

    Eager models are loaded during startup via ``preload``; others are
    loaded on first use via ``ensure`` (concurrent callers share a lock
    so the model is only loaded once).
    """

    def __init__(self) -> None:
        self._models: dict[str, FaceModel] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def get(self, name: str) -> FaceModel:
        return self._models[name]

    async def preload(self, name: str) -> FaceModel:
        self._models[name] = await asyncio.to_thread(_load_model, name)
        return self._models[name]

    async def ensure(self, name: str) -> FaceModel:
        if name in self._models:
            return self._models[name]
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            if name not in self._models:
                self._models[name] = await asyncio.to_thread(_load_model, name)
        return self._models[name]

    def clear(self) -> None:
        self._models.clear()


def _load_model(name: str) -> FaceModel:
    log.info("Loading %s (det_size=%s, ctx=cpu)...", name, DET_SIZE)
    started = time.time()
    try:
        model = insightface.app.FaceAnalysis(
            name=name,
            allowed_modules=["detection", "recognition"],
        )
    except AssertionError:
        # antelopev2.zip extracts into a nested directory that hides the
        # detection model from InsightFace's loader. Flatten and retry once.
        log.warning("%s failed to load (likely nested extraction); repairing and retrying", name)
        fix_antelopev2_nesting()
        model = insightface.app.FaceAnalysis(
            name=name,
            allowed_modules=["detection", "recognition"],
        )
    model.prepare(ctx_id=CTX_CPU, det_size=DET_SIZE)
    log.info("%s ready in %.1fs", name, time.time() - started)
    return model


def fix_antelopev2_nesting() -> None:
    """The antelopev2.zip extracts to a nested directory which breaks
    InsightFace's loader. Flatten it if necessary."""
    base = Path.home() / ".insightface" / "models" / "antelopev2"
    nested = base / "antelopev2"
    if nested.is_dir() and not (base / "scrfd_10g_bnkps.onnx").exists():
        for f in nested.iterdir():
            shutil.move(str(f), str(base / f.name))
        nested.rmdir()
        log.info("Fixed antelopev2 directory structure")
