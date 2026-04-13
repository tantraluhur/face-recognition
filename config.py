"""Application configuration and logging setup."""

from __future__ import annotations

import logging
import sys

# ── Model / matching ───────────────────────────────────────────────────
MATCH_THRESHOLD = 0.4
DET_SIZE = (640, 640)
CTX_CPU = -1  # InsightFace: ctx_id=-1 selects the CPU execution provider.

# Models loaded eagerly during startup. Others are loaded on first use.
EAGER_MODELS: tuple[str, ...] = ("buffalo_l",)

MODEL_CATALOG: dict[str, dict[str, object]] = {
    "buffalo_l": {
        "description": "SCRFD-10G + ResNet50 ArcFace",
        "size_mb": 182,
    },
    "antelopev2": {
        "description": "SCRFD-10G + ResNet100 ArcFace",
        "size_mb": 264,
    },
}


# ── Logging ────────────────────────────────────────────────────────────
def configure_logging() -> logging.Logger:
    """Install a sane default logging config and return the root app logger."""
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("face-recognition")
