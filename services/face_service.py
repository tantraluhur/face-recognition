"""Business logic: compare two face images with a loaded model."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from config import MATCH_THRESHOLD
from schemas import ModelResult
from services.model_registry import FaceModel


def _largest_face_embedding(model: FaceModel, img: np.ndarray) -> np.ndarray | None:
    """Return the L2-normalized embedding of the largest detected face."""
    faces = model.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


def _compare_sync(
    model: FaceModel,
    profile_img: np.ndarray,
    live_img: np.ndarray,
) -> ModelResult:
    start = time.time()

    def elapsed_ms() -> int:
        return int((time.time() - start) * 1000)

    profile_emb = _largest_face_embedding(model, profile_img)
    if profile_emb is None:
        return ModelResult(took_ms=elapsed_ms(), error="no_face_in_profile")

    live_emb = _largest_face_embedding(model, live_img)
    if live_emb is None:
        return ModelResult(took_ms=elapsed_ms(), error="no_face_in_live")

    similarity = float(np.dot(profile_emb, live_emb))
    return ModelResult(
        similarity=similarity,
        verified=similarity > MATCH_THRESHOLD,
        took_ms=elapsed_ms(),
    )


async def compare_faces(
    model: FaceModel,
    profile_img: np.ndarray,
    live_img: np.ndarray,
) -> ModelResult:
    """Run the blocking model inference off the event loop."""
    return await asyncio.to_thread(_compare_sync, model, profile_img, live_img)
