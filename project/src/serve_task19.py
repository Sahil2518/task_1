"""
Task 19 — FastAPI Predict Stub
PlaceMux Phase 1 Industry Immersion

Serves the versioned, serialised model+preprocessor artifact produced by
train_task19.py via a production-grade REST API.

Run with:
    cd project
    python -m src.serve_task19
or:
    uvicorn src.serve_task19:app --host 0.0.0.0 --port 8019 --reload
"""

import os
import json
import hashlib
import time
from typing import Optional, List

import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

# ── Paths (relative to project root) ─────────────────────────────────────────
TASK19_DIR    = os.path.join("models", "task19")
VERSION       = "1.0.0"
ARTIFACT_NAME = f"placemux_pipeline_v{VERSION}.joblib"
ARTIFACT_PATH = os.path.join(TASK19_DIR, ARTIFACT_NAME)
METADATA_PATH = os.path.join(TASK19_DIR, "metadata.json")

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class CandidateInput(BaseModel):
    """
    Input schema for a single candidate placement prediction.
    All fields go through Pydantic validation before hitting the model.
    """
    domain_score     : float = Field(..., ge=0.0, le=100.0,
                                     description="Domain knowledge score (0–100)")
    aptitude_score   : float = Field(..., ge=0.0, le=100.0,
                                     description="Aptitude test score (0–100)")
    projects_completed: int  = Field(..., ge=0, le=50,
                                     description="Number of projects completed")
    # Derived features — client may supply pre-computed values or let the API compute them
    active_days      : Optional[int]   = Field(None, ge=0,
                                               description="Days between registration and last login")
    score_diff       : Optional[float] = Field(None,
                                               description="domain_score minus aptitude_score (auto-computed if omitted)")
    engagement       : Optional[float] = Field(None, ge=0,
                                               description="active_days × projects_completed (auto-computed if omitted)")

    # Optional: if client passes raw dates we derive active_days ourselves
    registration_date: Optional[str] = Field(None, example="2025-01-15")
    last_login_date  : Optional[str] = Field(None, example="2025-06-01")

    @field_validator("domain_score", "aptitude_score")
    @classmethod
    def scores_must_be_finite(cls, v):
        import math
        if not math.isfinite(v):
            raise ValueError("Score must be a finite number.")
        return v


class BatchInput(BaseModel):
    """Batch prediction request — up to 100 candidates."""
    candidates: List[CandidateInput] = Field(..., min_length=1, max_length=100)


class PredictionResult(BaseModel):
    sample_index : int
    prediction   : int
    probability  : float
    is_placed    : bool
    confidence   : str    # "high" | "medium" | "low"


class SinglePredictionOutput(BaseModel):
    artifact_version : str
    prediction       : int
    probability      : float
    is_placed        : bool
    confidence       : str
    threshold        : float
    latency_ms       : float


class BatchPredictionOutput(BaseModel):
    artifact_version: str
    n_candidates    : int
    predictions     : List[PredictionResult]
    latency_ms      : float


class HealthOutput(BaseModel):
    status          : str
    artifact_version: str
    model_loaded    : bool
    artifact_sha256 : Optional[str]


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title      ="PlaceMux Task 19 — Model Serving API",
    description=(
        "REST API that loads the versioned, serialised model artifact produced "
        "by Task 19 (Application Model Serializing) and exposes predict endpoints "
        "with full Pydantic input validation."
    ),
    version    = VERSION,
    docs_url   = "/docs",
    redoc_url  = "/redoc",
)

# ── Global model state ────────────────────────────────────────────────────────

_pipeline       = None
_metadata       = {}
_artifact_sha   = None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_model():
    """Load the versioned artifact on startup with integrity check.

    Sets global _pipeline to None if anything goes wrong so endpoints
    can return a 503 with a useful message instead of crashing.
    """
    global _pipeline, _metadata, _artifact_sha

    if not os.path.exists(ARTIFACT_PATH):
        print(
            f"[WARNING] Artifact not found at '{ARTIFACT_PATH}'.\n"
            f"          Run: python -m src.train_task19"
        )
        return

    # ── Load & validate metadata sidecar ──
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH) as f:
                _metadata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Could not parse metadata.json: {e}. Skipping integrity check.")
            _metadata = {}

        expected_sha = _metadata.get("artifact", {}).get("sha256", "")
        try:
            _artifact_sha = _sha256_file(ARTIFACT_PATH)
        except (FileNotFoundError, IOError) as e:
            print(f"[ERROR] SHA-256 computation failed: {e}")
            return

        if expected_sha and expected_sha != _artifact_sha:
            print(
                f"[ERROR] SHA-256 mismatch!\n"
                f"  Expected : {expected_sha}\n"
                f"  Actual   : {_artifact_sha}\n"
                f"  Refusing to load — artifact may be corrupted or tampered with."
            )
            return  # _pipeline stays None → endpoints return 503

        print(f"[INFO] SHA-256 integrity check passed: {_artifact_sha[:16]}...")
    else:
        print("[WARNING] metadata.json not found. Skipping integrity check.")
        try:
            _artifact_sha = _sha256_file(ARTIFACT_PATH)
        except Exception:
            _artifact_sha = None

    # ── Load model ──
    try:
        _pipeline = joblib.load(ARTIFACT_PATH)
        if _pipeline is None:
            raise ValueError("joblib.load() returned None.")
        print(f"[INFO] Artifact loaded: {ARTIFACT_PATH} (v{VERSION})")
    except Exception as e:
        print(
            f"[ERROR] Failed to load artifact: {e}\n"
            f"  Possible cause: scikit-learn version mismatch between save and serve environments.\n"
            f"  Saved with: {_metadata.get('library_versions', {}).get('scikit_learn', 'unknown')}"
        )
        _pipeline = None


@app.on_event("startup")
def startup_event():
    _load_model()


# ── Middleware ────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    return response


# ── Feature engineering helper ────────────────────────────────────────────────

# Features the pipeline was trained on (mirrors src/data.py feature_engineering)
_REQUIRED_FEATURES = [
    "domain_score", "aptitude_score", "projects_completed",
    "active_days", "registration_month", "total_score",
]


def _prepare_features(candidate: CandidateInput) -> pd.DataFrame:
    """Convert Pydantic model → engineered feature DataFrame.

    Raises
    ------
    ValueError  – if a required field is missing and cannot be derived.
    """
    import math

    active_days       = candidate.active_days
    registration_month = None

    # ── Derive date-based features ──
    if active_days is None:
        if candidate.registration_date and candidate.last_login_date:
            try:
                reg  = pd.to_datetime(candidate.registration_date)
                last = pd.to_datetime(candidate.last_login_date)
                if last < reg:
                    raise ValueError(
                        f"last_login_date ({candidate.last_login_date}) is before "
                        f"registration_date ({candidate.registration_date})."
                    )
                active_days = max(0, (last - reg).days)
                registration_month = reg.month
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"Could not parse date fields: {e}") from e
        else:
            # Fallback: use 0 active days and month=1
            active_days = 0
            registration_month = 1
    else:
        # active_days was provided; derive month from registration_date if available
        if candidate.registration_date:
            try:
                reg = pd.to_datetime(candidate.registration_date)
                registration_month = reg.month
            except Exception:
                registration_month = 1
        else:
            registration_month = 1

    # ── Core score features ──
    domain_score  = candidate.domain_score
    aptitude_score = candidate.aptitude_score

    for name, val in [("domain_score", domain_score), ("aptitude_score", aptitude_score)]:
        if not math.isfinite(val):
            raise ValueError(f"'{name}' must be a finite number, got {val}.")

    total_score = domain_score + aptitude_score

    row = {
        "domain_score"      : domain_score,
        "aptitude_score"    : aptitude_score,
        "projects_completed": candidate.projects_completed,
        "active_days"       : active_days,
        "registration_month": registration_month,
        "total_score"       : total_score,
    }

    # Validate no required feature is missing
    missing = [k for k in _REQUIRED_FEATURES if row.get(k) is None]
    if missing:
        raise ValueError(f"Missing required features after engineering: {missing}")

    return pd.DataFrame([row])


def _confidence_label(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "high"
    if prob >= 0.60 or prob <= 0.40:
        return "medium"
    return "low"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthOutput, tags=["Monitoring"])
def health():
    """Returns API health and model load status."""
    return {
        "status"          : "healthy" if _pipeline is not None else "degraded",
        "artifact_version": VERSION,
        "model_loaded"    : _pipeline is not None,
        "artifact_sha256" : _artifact_sha,
    }


@app.post(
    "/v1/predict",
    response_model=SinglePredictionOutput,
    tags=["Prediction"],
    summary="Single candidate placement prediction",
)
def predict_single(input_data: CandidateInput, threshold: float = 0.40):
    """
    Predict placement for a single candidate.

    - **threshold**: Decision threshold (default 0.40 — business-optimal from Task 12).
    - Returns probability and binary placement decision.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded or failed integrity check. "
                "Run `python -m src.train_task19` to create the artifact, then restart."
            ),
        )

    if not (0.0 < threshold < 1.0):
        raise HTTPException(
            status_code=422,
            detail=f"threshold must be in (0, 1), got {threshold}.",
        )

    t0 = time.perf_counter()

    try:
        df = _prepare_features(input_data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected feature error: {e}")

    try:
        prob = float(_pipeline.predict_proba(df)[0][1])
        pred = int(prob >= threshold)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {e}. Check that the loaded artifact matches the current sklearn version.",
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "artifact_version": VERSION,
        "prediction"      : pred,
        "probability"     : round(prob, 4),
        "is_placed"       : pred == 1,
        "confidence"      : _confidence_label(prob),
        "threshold"       : threshold,
        "latency_ms"      : round(latency_ms, 3),
    }


@app.post(
    "/v1/predict/batch",
    response_model=BatchPredictionOutput,
    tags=["Prediction"],
    summary="Batch candidate placement prediction (up to 100)",
)
def predict_batch(input_data: BatchInput, threshold: float = 0.40):
    """
    Predict placement for a batch of up to 100 candidates in one request.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded or failed integrity check. "
                "Run `python -m src.train_task19` to create the artifact, then restart."
            ),
        )

    if not (0.0 < threshold < 1.0):
        raise HTTPException(
            status_code=422,
            detail=f"threshold must be in (0, 1), got {threshold}.",
        )

    if len(input_data.candidates) == 0:
        raise HTTPException(status_code=422, detail="candidates list is empty.")

    t0 = time.perf_counter()

    # Feature engineering — collect per-candidate errors rather than failing the whole batch
    dfs, failed_indices = [], []
    for i, c in enumerate(input_data.candidates):
        try:
            dfs.append(_prepare_features(c))
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Feature engineering error for candidate[{i}]: {e}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error for candidate[{i}]: {e}",
            )

    try:
        df    = pd.concat(dfs, ignore_index=True)
        probs = _pipeline.predict_proba(df)[:, 1]
        preds = (probs >= threshold).astype(int)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch inference failed: {e}. Check artifact/sklearn version compatibility.",
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    predictions = [
        {
            "sample_index": i,
            "prediction"  : int(p),
            "probability" : round(float(pr), 4),
            "is_placed"   : bool(p == 1),
            "confidence"  : _confidence_label(float(pr)),
        }
        for i, (p, pr) in enumerate(zip(preds, probs))
    ]

    return {
        "artifact_version": VERSION,
        "n_candidates"    : len(predictions),
        "predictions"     : predictions,
        "latency_ms"      : round(latency_ms, 3),
    }


@app.get("/v1/metadata", tags=["Monitoring"])
def get_metadata():
    """Return the full metadata sidecar for the loaded artifact."""
    if not _metadata:
        raise HTTPException(status_code=404, detail="Metadata not found.")
    return _metadata


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serve_task19:app",
        host="0.0.0.0",
        port=8019,
        reload=True,
        reload_dirs=["src"],
    )
