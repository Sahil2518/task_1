"""
Task 20 — End-to-End Pipelines & Deployment (Flask)
PlaceMux Phase 1 Industry Immersion · Capstone

Loads the versioned, serialised artifact from Task 19 and exposes:
  GET  /health           — liveness + model-load status
  POST /v1/predict       — single candidate placement prediction
  POST /v1/predict/batch — batch prediction (up to 100 candidates)
  GET  /v1/metadata      — full artifact metadata sidecar

Run from project root:
    python -m src.serve_task20
    -- or --
    python src/serve_task20.py
"""

import os
import sys
import json
import hashlib
import logging
import math
import time
import datetime

import joblib
import pandas as pd
from flask import Flask, request, jsonify, g

# ── Logging setup ─────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task20.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Artifact paths (relative to project root) ─────────────────────────────────

TASK19_DIR    = os.path.join("models", "task19")
VERSION       = "1.0.0"
ARTIFACT_NAME = f"placemux_pipeline_v{VERSION}.joblib"
ARTIFACT_PATH = os.path.join(TASK19_DIR, ARTIFACT_NAME)
METADATA_PATH = os.path.join(TASK19_DIR, "metadata.json")

# Features the pipeline was trained on (mirrors src/data.py)
REQUIRED_FEATURES = [
    "domain_score", "aptitude_score", "projects_completed",
    "active_days", "registration_month", "total_score",
]

# ── Global model state ─────────────────────────────────────────────────────────

_pipeline     = None   # loaded sklearn pipeline
_metadata     = {}     # metadata sidecar dict
_artifact_sha = None   # SHA-256 of the artifact file


# ── Helper: SHA-256 ───────────────────────────────────────────────────────────

def _sha256_file(path: str) -> str:
    """Return hex SHA-256 digest of the file at *path*.

    Raises
    ------
    FileNotFoundError – path does not exist.
    IOError           – file cannot be read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except (IOError, OSError) as exc:
        raise IOError(f"Cannot read file for SHA-256: {path} — {exc}") from exc
    return h.hexdigest()


# ── Model loader (called once at startup) ─────────────────────────────────────

def _load_model() -> None:
    """Load and integrity-check the Task 19 joblib artifact.

    Sets global _pipeline / _metadata / _artifact_sha.
    On any failure, _pipeline is left as None so endpoints
    return 503 with a clear message rather than crashing.
    """
    global _pipeline, _metadata, _artifact_sha

    logger.info("=" * 60)
    logger.info("  Task 20 — Flask Deployment Service starting up")
    logger.info("=" * 60)

    # ── Artifact existence ──
    if not os.path.exists(ARTIFACT_PATH):
        logger.warning(
            "Artifact not found at '%s'. "
            "Run: python -m src.train_task19  — then restart.", ARTIFACT_PATH
        )
        return

    # ── Metadata sidecar ──
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH) as f:
                _metadata = json.load(f)
            logger.info("Metadata loaded from %s", METADATA_PATH)
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Could not parse metadata.json (%s). Skipping integrity check.", exc)
            _metadata = {}
    else:
        logger.warning("metadata.json not found. Skipping integrity check.")

    # ── SHA-256 integrity check ──
    try:
        _artifact_sha = _sha256_file(ARTIFACT_PATH)
    except (FileNotFoundError, IOError) as exc:
        logger.error("SHA-256 computation failed: %s", exc)
        return

    expected_sha = _metadata.get("artifact", {}).get("sha256", "")
    if expected_sha and expected_sha != _artifact_sha:
        logger.error(
            "SHA-256 MISMATCH — refusing to load.\n"
            "  Expected : %s\n  Actual   : %s\n"
            "  Artifact may be corrupted or tampered with.",
            expected_sha, _artifact_sha,
        )
        return  # _pipeline stays None → endpoints return 503

    if expected_sha:
        logger.info("SHA-256 integrity check PASSED (%s...)", _artifact_sha[:16])
    else:
        logger.info("No expected SHA-256 in metadata — skipping comparison.")

    # ── Load pipeline ──
    try:
        _pipeline = joblib.load(ARTIFACT_PATH)
        if _pipeline is None:
            raise ValueError("joblib.load() returned None.")
        logger.info("Artifact loaded successfully: %s (v%s)", ARTIFACT_PATH, VERSION)
    except Exception as exc:
        saved_sk = _metadata.get("library_versions", {}).get("scikit_learn", "unknown")
        logger.error(
            "Failed to load artifact: %s\n"
            "  Possible cause: scikit-learn version mismatch "
            "(artifact saved with v%s).",
            exc, saved_sk,
        )
        _pipeline = None


# ── Feature engineering ───────────────────────────────────────────────────────

def _prepare_features(data: dict) -> pd.DataFrame:
    """Validate and engineer features from a raw request dict.

    Parameters
    ----------
    data : dict  Raw JSON body for one candidate.

    Returns
    -------
    pd.DataFrame  One-row DataFrame with exactly REQUIRED_FEATURES columns.

    Raises
    ------
    ValueError  On missing/invalid fields.
    """
    errors = []

    # ── Required numeric fields ──
    domain_score = data.get("domain_score")
    aptitude_score = data.get("aptitude_score")
    projects_completed = data.get("projects_completed")

    if domain_score is None:
        errors.append("'domain_score' is required.")
    elif not isinstance(domain_score, (int, float)) or not math.isfinite(domain_score):
        errors.append("'domain_score' must be a finite number.")
    elif not (0.0 <= domain_score <= 100.0):
        errors.append("'domain_score' must be in [0, 100].")

    if aptitude_score is None:
        errors.append("'aptitude_score' is required.")
    elif not isinstance(aptitude_score, (int, float)) or not math.isfinite(aptitude_score):
        errors.append("'aptitude_score' must be a finite number.")
    elif not (0.0 <= aptitude_score <= 100.0):
        errors.append("'aptitude_score' must be in [0, 100].")

    if projects_completed is None:
        errors.append("'projects_completed' is required.")
    elif not isinstance(projects_completed, int) or projects_completed < 0:
        errors.append("'projects_completed' must be a non-negative integer.")

    if errors:
        raise ValueError("; ".join(errors))

    # ── Optional / derived fields ──
    active_days = data.get("active_days", 0)
    if not isinstance(active_days, (int, float)) or active_days < 0:
        raise ValueError("'active_days' must be a non-negative number.")
    active_days = int(active_days)

    # Derive registration_month from registration_date if provided
    registration_month = data.get("registration_month", 1)
    registration_date = data.get("registration_date")
    if registration_date:
        try:
            registration_month = pd.to_datetime(registration_date).month
        except Exception:
            raise ValueError(f"Cannot parse 'registration_date': '{registration_date}'.")

    # Derive active_days from date pair if both provided
    last_login_date = data.get("last_login_date")
    if registration_date and last_login_date:
        try:
            reg  = pd.to_datetime(registration_date)
            last = pd.to_datetime(last_login_date)
            if last < reg:
                raise ValueError(
                    f"'last_login_date' ({last_login_date}) is before "
                    f"'registration_date' ({registration_date})."
                )
            active_days = max(0, (last - reg).days)
            registration_month = reg.month
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Cannot parse date fields: {exc}") from exc

    total_score = float(domain_score) + float(aptitude_score)

    row = {
        "domain_score"      : float(domain_score),
        "aptitude_score"    : float(aptitude_score),
        "projects_completed": int(projects_completed),
        "active_days"       : int(active_days),
        "registration_month": int(registration_month),
        "total_score"       : total_score,
    }

    # Final guard: ensure all required features present
    missing = [k for k in REQUIRED_FEATURES if row.get(k) is None]
    if missing:
        raise ValueError(f"Missing required features after engineering: {missing}")

    return pd.DataFrame([row])


def _confidence_label(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "high"
    if prob >= 0.60 or prob <= 0.40:
        return "medium"
    return "low"


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ── Latency middleware ────────────────────────────────────────────────────────

@app.before_request
def _start_timer():
    g.t0 = time.perf_counter()


@app.after_request
def _inject_latency_header(response):
    latency_ms = (time.perf_counter() - g.t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    return response


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(exc):
    logger.warning("400 Bad Request: %s", exc)
    return jsonify({"error": "bad_request", "detail": str(exc)}), 400


@app.errorhandler(404)
def not_found(exc):
    return jsonify({"error": "not_found", "detail": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(exc):
    return jsonify({"error": "method_not_allowed", "detail": str(exc)}), 405


@app.errorhandler(500)
def internal_error(exc):
    logger.error("500 Internal Server Error: %s", exc, exc_info=True)
    return jsonify({"error": "internal_server_error", "detail": "An unexpected error occurred."}), 500


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness + model-load status check."""
    status = "healthy" if _pipeline is not None else "degraded"
    logger.info("GET /health  →  %s", status)
    return jsonify({
        "status"          : status,
        "artifact_version": VERSION,
        "model_loaded"    : _pipeline is not None,
        "artifact_sha256" : _artifact_sha,
        "timestamp"       : datetime.datetime.utcnow().isoformat() + "Z",
    }), 200


@app.route("/v1/predict", methods=["POST"])
def predict_single():
    """
    Single-candidate placement prediction.

    Body (JSON):
        domain_score       float  [0, 100]  — required
        aptitude_score     float  [0, 100]  — required
        projects_completed int    >= 0      — required
        active_days        int    >= 0      — optional (default 0)
        registration_month int    1-12      — optional (default 1)
        registration_date  str    ISO date  — optional
        last_login_date    str    ISO date  — optional
        threshold          float  (0, 1)    — optional (default 0.40)
    """
    if _pipeline is None:
        return jsonify({
            "error" : "model_unavailable",
            "detail": (
                "Model not loaded. "
                "Run `python -m src.train_task19` to create the artifact, then restart."
            ),
        }), 503

    body = request.get_json(silent=True)
    if not body or not isinstance(body, dict):
        return jsonify({
            "error" : "bad_request",
            "detail": "Request body must be a JSON object.",
        }), 400

    # Threshold
    threshold = body.get("threshold", 0.40)
    try:
        threshold = float(threshold)
        if not (0.0 < threshold < 1.0):
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({
            "error" : "bad_request",
            "detail": f"'threshold' must be a float in (0, 1), got: {threshold}",
        }), 400

    # Feature engineering + validation
    t0 = time.perf_counter()
    try:
        df = _prepare_features(body)
    except ValueError as exc:
        logger.warning("POST /v1/predict — validation error: %s", exc)
        return jsonify({"error": "validation_error", "detail": str(exc)}), 422
    except Exception as exc:
        logger.error("POST /v1/predict — feature engineering error: %s", exc, exc_info=True)
        return jsonify({"error": "internal_error", "detail": str(exc)}), 500

    # Inference
    try:
        prob = float(_pipeline.predict_proba(df)[0][1])
        pred = int(prob >= threshold)
    except Exception as exc:
        logger.error("POST /v1/predict — inference error: %s", exc, exc_info=True)
        return jsonify({
            "error" : "inference_error",
            "detail": f"Model inference failed: {exc}",
        }), 500

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    result = {
        "artifact_version": VERSION,
        "prediction"      : pred,
        "probability"     : round(prob, 4),
        "is_placed"       : pred == 1,
        "confidence"      : _confidence_label(prob),
        "threshold"       : threshold,
        "latency_ms"      : latency_ms,
    }
    logger.info("POST /v1/predict  →  placed=%s  prob=%.4f  latency=%.1fms",
                pred, prob, latency_ms)
    return jsonify(result), 200


@app.route("/v1/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch placement prediction for up to 100 candidates.

    Body (JSON):
        { "candidates": [ <candidate_object>, ... ] }
        threshold  float  optional, default 0.40
    """
    if _pipeline is None:
        return jsonify({
            "error" : "model_unavailable",
            "detail": "Model not loaded. Run train_task19 first.",
        }), 503

    body = request.get_json(silent=True)
    if not body or not isinstance(body, dict):
        return jsonify({"error": "bad_request", "detail": "Body must be a JSON object."}), 400

    candidates = body.get("candidates")
    if not isinstance(candidates, list) or len(candidates) == 0:
        return jsonify({
            "error" : "validation_error",
            "detail": "'candidates' must be a non-empty list.",
        }), 422

    if len(candidates) > 100:
        return jsonify({
            "error" : "validation_error",
            "detail": f"Max 100 candidates per batch, got {len(candidates)}.",
        }), 422

    threshold = body.get("threshold", 0.40)
    try:
        threshold = float(threshold)
        if not (0.0 < threshold < 1.0):
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({
            "error" : "bad_request",
            "detail": f"'threshold' must be in (0, 1), got: {threshold}",
        }), 400

    t0 = time.perf_counter()

    # Per-candidate feature engineering
    dfs = []
    for i, c in enumerate(candidates):
        if not isinstance(c, dict):
            return jsonify({
                "error" : "validation_error",
                "detail": f"candidates[{i}] must be a JSON object.",
            }), 422
        try:
            dfs.append(_prepare_features(c))
        except ValueError as exc:
            return jsonify({
                "error" : "validation_error",
                "detail": f"candidates[{i}]: {exc}",
            }), 422
        except Exception as exc:
            logger.error("Batch feature error at index %d: %s", i, exc, exc_info=True)
            return jsonify({
                "error" : "internal_error",
                "detail": f"candidates[{i}]: unexpected error — {exc}",
            }), 500

    # Batch inference
    try:
        df    = pd.concat(dfs, ignore_index=True)
        probs = _pipeline.predict_proba(df)[:, 1]
        preds = (probs >= threshold).astype(int)
    except Exception as exc:
        logger.error("Batch inference failed: %s", exc, exc_info=True)
        return jsonify({
            "error" : "inference_error",
            "detail": f"Batch model inference failed: {exc}",
        }), 500

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

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

    logger.info("POST /v1/predict/batch  →  n=%d  latency=%.1fms",
                len(predictions), latency_ms)
    return jsonify({
        "artifact_version": VERSION,
        "n_candidates"    : len(predictions),
        "predictions"     : predictions,
        "latency_ms"      : latency_ms,
    }), 200


@app.route("/v1/metadata", methods=["GET"])
def get_metadata():
    """Return the full artifact metadata sidecar."""
    if not _metadata:
        return jsonify({"error": "not_found", "detail": "Metadata not available."}), 404
    return jsonify(_metadata), 200


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _load_model()
    logger.info("Starting Flask server on http://0.0.0.0:5020")
    logger.info("Endpoints: GET /health  POST /v1/predict  POST /v1/predict/batch  GET /v1/metadata")
    app.run(host="0.0.0.0", port=5020, debug=False)


if __name__ == "__main__":
    main()
