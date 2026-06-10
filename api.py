"""
api.py — FakeShield AI FastAPI REST Server
==========================================
Exposes the inference engine as a JSON REST API.

Start with:
    uvicorn api:app --host 0.0.0.0 --port 8000
    # or via main.py:
    python main.py --web --mode api --port 8000

Endpoints
---------
GET  /              → redirect to /docs
GET  /health        → liveness + readiness check
POST /predict       → multimodal fake-news analysis
POST /predict/text  → text-only shortcut
POST /predict/image → image-only shortcut
GET  /history       → recent predictions from SQLite
DELETE /history     → clear history

File size limits
----------------
Images : 10 MB
Videos : see AppConfig.VIDEO_MAX_DURATION (we don't upload video — only URL)
"""

from __future__ import annotations

import io
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, List, Optional

# ── Ensure project root importable ──────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import AppConfig

# ── FastAPI ──────────────────────────────────────────────────────────────────
try:
    from fastapi import (
        BackgroundTasks, Depends, FastAPI, File, Form,
        HTTPException, Request, UploadFile, status,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, RedirectResponse
    from pydantic import BaseModel, Field, field_validator
except ImportError as e:
    raise SystemExit(f"FastAPI not installed: {e}  →  pip install fastapi[all]")

AppConfig.ensure_dirs()

logger = logging.getLogger("api")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)

# ════════════════════════════════════════════════════════════════════════════
# App
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="FakeShield AI",
    description=(
        "Multimodal Fake News Detection System — "
        "analyses text, images and videos for credibility."
    ),
    version=AppConfig.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Simple in-memory rate limiter (per IP, requests per minute) ──────────────
_RATE_STORE: dict[str, list[float]] = {}
_RATE_LIMIT  = 20   # requests
_RATE_WINDOW = 60   # seconds

def _check_rate_limit(request: Request):
    ip    = request.client.host if request.client else "unknown"
    now   = time.time()
    hits  = _RATE_STORE.get(ip, [])
    hits  = [t for t in hits if now - t < _RATE_WINDOW]
    if len(hits) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {_RATE_LIMIT} requests / {_RATE_WINDOW}s",
        )
    hits.append(now)
    _RATE_STORE[ip] = hits


# ── Lazy inference engine ────────────────────────────────────────────────────
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from inference import FakeNewsInference
        _engine = FakeNewsInference.get_instance()
    return _engine


# ════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ════════════════════════════════════════════════════════════════════════════

class TextRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10_000,
                      description="Article text or URL")
    save_to_db: bool = Field(True, description="Persist result to SQLite")


class PredictionResult(BaseModel):
    request_id:  str
    verdict:     str
    final_score: float
    confidence:  float
    individual:  dict
    explanations:dict
    modalities:  list
    meta:        dict
    elapsed_ms:  float


class HealthResponse(BaseModel):
    status:      str
    version:     str
    models_ready:dict
    uptime_sec:  float


class HistoryItem(BaseModel):
    id:            int
    timestamp:     str
    analysis_type: str
    verdict:       str
    fake_score:    float
    confidence:    float
    summary:       str


# ════════════════════════════════════════════════════════════════════════════
# Startup event
# ════════════════════════════════════════════════════════════════════════════

_START_TIME = time.time()

@app.on_event("startup")
async def _startup():
    logger.info("FakeShield AI API starting up …")


# ════════════════════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness + readiness probe."""
    engine = get_engine()
    return HealthResponse(
        status     = "ok",
        version    = AppConfig.APP_VERSION,
        models_ready={
            "text":  engine._text_analyzer  is not None,
            "image": engine._image_analyzer is not None,
            "video": engine._video_analyzer is not None,
        },
        uptime_sec = round(time.time() - _START_TIME, 1),
    )


# ── Full multimodal prediction ────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResult,
    tags=["Inference"],
    summary="Analyse text + optional image + optional video URL",
)
async def predict(
    request:          Request,
    background_tasks: BackgroundTasks,
    text:       Optional[str]        = Form(None, description="Article text or URL"),
    video_url:  Optional[str]        = Form(None, description="Video URL"),
    image_file: Optional[UploadFile] = File(None, description="Image file ≤10 MB"),
    save_to_db: bool                 = Form(True),
    _rate:      None                 = Depends(_check_rate_limit),
):
    """
    **Multimodal fake-news analysis.**

    - `text`        : paste article text **or** a URL (automatically scraped).
    - `image_file`  : optional image upload (JPEG / PNG / WebP, ≤ 10 MB).
    - `video_url`   : optional YouTube / direct .mp4 URL.
    - `save_to_db`  : persist result to SQLite (default `true`).

    Returns a `PredictionResult` JSON.
    """
    if not text and not image_file and not video_url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one of: text, image_file, video_url",
        )

    req_id = str(uuid.uuid4())[:8]
    t0     = time.perf_counter()

    # ── Resolve image ────────────────────────────────────────────────────────
    pil_image = None
    if image_file:
        _validate_image_upload(image_file)
        pil_image = _read_uploaded_image(await image_file.read())

    # ── Run inference ────────────────────────────────────────────────────────
    engine = get_engine()
    try:
        result = engine.predict(
            text       = text,
            image      = pil_image,
            video_url  = video_url,
            save_to_db = save_to_db,
        )
    except Exception as exc:
        logger.exception("Inference failed for request %s", req_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis error: {exc}",
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return PredictionResult(
        request_id   = req_id,
        verdict      = result["verdict"],
        final_score  = result["final_score"],
        confidence   = result["confidence"],
        individual   = result.get("individual", {}),
        explanations = result.get("explanations", {}),
        modalities   = result.get("modalities", []),
        meta         = {**result.get("meta", {}),
                        "request_id": req_id},
        elapsed_ms   = elapsed_ms,
    )


# ── Text-only shortcut ────────────────────────────────────────────────────────
@app.post(
    "/predict/text",
    response_model=PredictionResult,
    tags=["Inference"],
    summary="Text-only analysis (JSON body)",
)
async def predict_text(
    request: Request,
    body:    TextRequest,
    _rate:   None = Depends(_check_rate_limit),
):
    """Analyse article text or URL. Accepts JSON body."""
    t0     = time.perf_counter()
    req_id = str(uuid.uuid4())[:8]
    engine = get_engine()

    try:
        result = engine.predict(
            text       = body.text,
            save_to_db = body.save_to_db,
        )
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return PredictionResult(
        request_id   = req_id,
        verdict      = result["verdict"],
        final_score  = result["final_score"],
        confidence   = result["confidence"],
        individual   = result.get("individual", {}),
        explanations = result.get("explanations", {}),
        modalities   = result.get("modalities", []),
        meta         = result.get("meta", {}),
        elapsed_ms   = elapsed_ms,
    )


# ── Image-only shortcut ───────────────────────────────────────────────────────
@app.post(
    "/predict/image",
    response_model=PredictionResult,
    tags=["Inference"],
    summary="Image-only analysis",
)
async def predict_image(
    request:    Request,
    image_file: UploadFile = File(..., description="JPEG / PNG / WebP ≤ 10 MB"),
    save_to_db: bool       = Form(True),
    _rate:      None       = Depends(_check_rate_limit),
):
    """Upload an image for manipulation / deepfake analysis."""
    _validate_image_upload(image_file)
    pil_image  = _read_uploaded_image(await image_file.read())
    t0         = time.perf_counter()
    req_id     = str(uuid.uuid4())[:8]
    engine     = get_engine()

    try:
        result = engine.predict(image=pil_image, save_to_db=save_to_db)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return PredictionResult(
        request_id   = req_id,
        verdict      = result["verdict"],
        final_score  = result["final_score"],
        confidence   = result["confidence"],
        individual   = result.get("individual", {}),
        explanations = result.get("explanations", {}),
        modalities   = result.get("modalities", []),
        meta         = result.get("meta", {}),
        elapsed_ms   = elapsed_ms,
    )


# ── History ───────────────────────────────────────────────────────────────────
@app.get(
    "/history",
    response_model=List[HistoryItem],
    tags=["History"],
    summary="List recent analyses",
)
async def get_history(limit: int = 50):
    """Return the most recent *limit* analysis records from SQLite."""
    from database import AnalysisDatabase
    db = AnalysisDatabase()
    try:
        records = db.get_history(limit)
        return [
            HistoryItem(
                id            = r["id"],
                timestamp     = r["timestamp"],
                analysis_type = r.get("analysis_type", "?"),
                verdict       = r.get("overall_label", "?"),
                fake_score    = r.get("overall_score", 0.5),
                confidence    = r.get("confidence", 0.0),
                summary       = (r.get("input_summary") or "")[:80],
            )
            for r in records
        ]
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


@app.delete(
    "/history",
    tags=["History"],
    summary="Clear all analysis history",
)
async def clear_history():
    """Wipe the analysis history from SQLite."""
    from database import AnalysisDatabase
    AnalysisDatabase().clear_all()
    return {"status": "cleared"}


# ── Statistics ────────────────────────────────────────────────────────────────
@app.get("/stats", tags=["History"], summary="Aggregate statistics")
async def get_stats():
    from database import AnalysisDatabase
    return AnalysisDatabase().get_statistics()


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════

_MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB
_ALLOWED_MIME    = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def _validate_image_upload(file: UploadFile):
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {file.content_type}. "
                   f"Accepted: {', '.join(_ALLOWED_MIME)}",
        )


def _read_uploaded_image(raw: bytes):
    if len(raw) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large ({len(raw)/1e6:.1f} MB). Max: 10 MB.",
        )
    try:
        from PIL import Image
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image: {exc}",
        )


# ════════════════════════════════════════════════════════════════════════════
# Standalone run (python api.py)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = False,
        log_level = "info",
    )
