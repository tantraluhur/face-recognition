"""POST /api/verify — compare two face images with both models in parallel."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from schemas import VerifyRequest, VerifyResponse
from services.face_service import compare_faces
from services.model_registry import ModelRegistry
from utils.image import decode_base64_image

router = APIRouter(prefix="/api", tags=["verify"])
log = logging.getLogger(__name__)


@router.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest, request: Request) -> VerifyResponse:
    try:
        profile_img = decode_base64_image(req.profile_image)
        live_img = decode_base64_image(req.live_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    registry: ModelRegistry = request.app.state.models
    antelopev2 = await registry.ensure("antelopev2")

    buffalo_result, antelopev2_result = await asyncio.gather(
        compare_faces(registry.get("buffalo_l"), profile_img, live_img),
        compare_faces(antelopev2, profile_img, live_img),
    )

    log.info("verify: buffalo_l=%s antelopev2=%s", buffalo_result, antelopev2_result)

    return VerifyResponse(buffalo_l=buffalo_result, antelopev2=antelopev2_result)
