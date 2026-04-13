"""Image decoding helpers."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def decode_base64_image(b64: str) -> np.ndarray:
    """Decode a (possibly data-URI-prefixed) base64 image into a BGR ndarray.

    InsightFace/OpenCV expect BGR, while PIL produces RGB — hence the flip.
    """
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img)[:, :, ::-1].copy()
