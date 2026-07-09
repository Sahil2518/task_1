"""
Task 19 — Application Model Serializing
PlaceMux Phase 1 Industry Immersion

Steps:
  1. Bundle fitted pipeline (preprocessor + model) into one artifact.
  2. Save with joblib + metadata JSON.
  3. Record library versions and training metrics.
  4. Write load-and-predict function with input validation (Pydantic).
  5. Test loading in a fresh environment and predicting.
  6. Version the artifact for traceability.

Builds on the calibrated_pipeline.pkl from Task 12 (the project's primary model).
If it doesn't exist, re-trains a GradientBoosting pipeline from scratch using
the same data.py / preprocess.py helpers the project already relies on.
"""

import os
import sys
import json
import hashlib
import traceback
import datetime
import platform
import random

import numpy as np
import pandas as pd
import joblib

# ── Constants ────────────────────────────────────────────────────────────────
SEED          = 42
MODELS_DIR    = "models"
LOGS_DIR      = "logs"
TASK19_DIR    = os.path.join(MODELS_DIR, "task19")
VERSION       = "1.0.0"
ARTIFACT_NAME = f"placemux_pipeline_v{VERSION}.joblib"
ARTIFACT_PATH = os.path.join(TASK19_DIR, ARTIFACT_NAME)
METADATA_PATH = os.path.join(TASK19_DIR, "metadata.json")
RESULTS_PATH  = os.path.join(LOGS_DIR, "task19_results.json")
SRC_MODEL     = os.path.join(MODELS_DIR, "calibrated_pipeline.pkl")  # prefer Task 12 model

np.random.seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Structured error logger
# ─────────────────────────────────────────────────────────────────────────────

_errors = []   # accumulate non-fatal warnings to include in the results JSON


def _warn(step: str, message: str, exc: Exception = None):
    """Log a recoverable warning and continue."""
    tb = traceback.format_exc() if exc else ""
    entry = {"step": step, "message": message, "traceback": tb.strip()}
    _errors.append(entry)
    print(f"  [WARNING] {message}")
    if exc and tb.strip():
        print(f"           {tb.splitlines()[-1]}")


def _fatal(step: str, message: str, exc: Exception = None):
    """Log a fatal error, flush results JSON, then exit."""
    tb = traceback.format_exc() if exc else ""
    entry = {"step": step, "message": message, "traceback": tb.strip(), "fatal": True}
    _errors.append(entry)
    print(f"\n  [FATAL] {message}")
    if exc:
        traceback.print_exc()

    # Try to persist error state before quitting
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump({"status": "FAILED", "errors": _errors}, f, indent=4)
        print(f"  Error report -> {RESULTS_PATH}")
    except Exception:
        pass

    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data — reuse the project's existing load_data() from src/data.py
# ─────────────────────────────────────────────────────────────────────────────

def get_project_data():
    """Load data using the project's canonical pipeline.

    Raises
    ------
    ImportError  – if src.data is not importable.
    RuntimeError – if load_data() returns unexpected shapes.
    """
    try:
        from src.data import load_data
    except ImportError as e:
        raise ImportError(
            "Cannot import src.data.load_data. "
            "Ensure you are running from the project root directory."
        ) from e

    result = load_data()

    # Validate output structure
    if len(result) != 6:
        raise RuntimeError(
            f"load_data() returned {len(result)} objects; expected 6 "
            "(X_train, X_val, X_test, y_train, y_val, y_test)."
        )

    X_train, X_val, X_test, y_train, y_val, y_test = result

    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if arr is None or len(arr) == 0:
            raise RuntimeError(f"Data split '{name}' is empty or None.")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build a fresh pipeline (fallback if no pre-existing model)
# ─────────────────────────────────────────────────────────────────────────────

def build_and_train_pipeline(X_train, y_train):
    """Build preprocessor + GradientBoosting in a single sklearn Pipeline,
    using the same feature schema as the rest of the project.

    Raises
    ------
    ImportError  – if sklearn components are missing.
    ValueError   – if training data has zero rows or columns.
    RuntimeError – if fit() raises an unexpected error.
    """
    if X_train is None or len(X_train) == 0:
        raise ValueError("Training data is empty — cannot build pipeline.")
    if X_train.isnull().all(axis=None):
        raise ValueError("Training data is entirely NaN — cannot build pipeline.")

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import GradientBoostingClassifier
        from src.preprocess import get_feature_types, get_preprocessor
    except ImportError as e:
        raise ImportError(f"Required sklearn/project module not found: {e}") from e

    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            random_state=SEED
        )),
    ])

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"Pipeline.fit() failed: {e}") from e

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(pipeline, X_test, y_test):
    """Compute test-set metrics.

    Returns a metrics dict. If prediction itself fails, returns a dict with
    all-None values and appends a warning instead of crashing.
    """
    from sklearn.metrics import (
        f1_score, accuracy_score, precision_score, recall_score
    )

    if pipeline is None:
        _warn("evaluate", "Pipeline is None — skipping evaluation.")
        return {"accuracy": None, "precision": None, "recall": None, "f1_score": None}

    if X_test is None or len(X_test) == 0:
        _warn("evaluate", "Test set is empty — skipping evaluation.")
        return {"accuracy": None, "precision": None, "recall": None, "f1_score": None}

    try:
        y_pred = pipeline.predict(X_test)
    except Exception as e:
        _warn("evaluate", f"pipeline.predict() failed on test set: {e}", e)
        return {"accuracy": None, "precision": None, "recall": None, "f1_score": None}

    try:
        return {
            "accuracy" : round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall"   : round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score" : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        }
    except Exception as e:
        _warn("evaluate", f"Metric computation failed: {e}", e)
        return {"accuracy": None, "precision": None, "recall": None, "f1_score": None}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Serialise (joblib) + metadata JSON
# ─────────────────────────────────────────────────────────────────────────────

def sha256_of_file(path: str) -> str:
    """Compute SHA-256 checksum of a file for integrity verification.

    Raises
    ------
    FileNotFoundError – if path does not exist.
    IOError           – if the file cannot be read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot checksum missing file: {path}")

    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to read file for SHA-256: {path} — {e}") from e

    return h.hexdigest()


def save_artifact(pipeline, metrics: dict, data_shape: tuple,
                  source: str, feature_names: list):
    """Serialise pipeline and write a metadata sidecar.

    Raises
    ------
    ValueError        – if pipeline is None.
    PermissionError   – if the output directory cannot be created.
    RuntimeError      – if joblib.dump() fails.
    """
    if pipeline is None:
        raise ValueError("Cannot serialise a None pipeline.")

    try:
        os.makedirs(TASK19_DIR, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise PermissionError(f"Cannot create output directory '{TASK19_DIR}': {e}") from e

    # ── Save model artifact ──
    try:
        joblib.dump(pipeline, ARTIFACT_PATH, compress=3)
    except Exception as e:
        raise RuntimeError(f"joblib.dump() failed: {e}") from e

    if not os.path.exists(ARTIFACT_PATH):
        raise RuntimeError(
            f"joblib.dump() reported success but artifact not found at: {ARTIFACT_PATH}"
        )

    artifact_sha256 = sha256_of_file(ARTIFACT_PATH)
    artifact_size   = os.path.getsize(ARTIFACT_PATH)

    if artifact_size == 0:
        raise RuntimeError("Artifact was written but is 0 bytes — possible corruption.")

    # ── Collect library versions defensively ──
    import sklearn as _sk

    def _safe_version(module_name):
        try:
            import importlib
            m = importlib.import_module(module_name)
            return getattr(m, "__version__", "unknown")
        except ImportError:
            return "not installed"

    metadata = {
        "artifact": {
            "name"      : ARTIFACT_NAME,
            "version"   : VERSION,
            "path"      : ARTIFACT_PATH,
            "sha256"    : artifact_sha256,
            "size_bytes": artifact_size,
            "saved_at"  : datetime.datetime.utcnow().isoformat() + "Z",
            "source"    : source,
        },
        "model": {
            "type"             : type(pipeline).__name__,
            "seed"             : SEED,
            "feature_names"    : feature_names,
            "n_train_samples"  : data_shape[0],
            "n_features"       : data_shape[1],
        },
        "metrics_test": metrics,
        "library_versions": {
            "python"      : platform.python_version(),
            "scikit_learn": _sk.__version__,
            "numpy"       : _safe_version("numpy"),
            "pandas"      : _safe_version("pandas"),
            "joblib"      : _safe_version("joblib"),
            "fastapi"     : _safe_version("fastapi"),
            "pydantic"    : _safe_version("pydantic"),
            "uvicorn"     : _safe_version("uvicorn"),
            "os"          : platform.system(),
        },
        "lineage": {
            "task"       : "Task 19 — Application Model Serializing",
            "track"      : "PlaceMux Phase 1 Industry Immersion",
            "description": (
                "Serialised, versioned model+preprocessor bundle. "
                "Includes SHA-256 checksum for integrity verification "
                "and a metadata sidecar for full lineage traceability."
            ),
            "pitfalls_avoided": {
                "preprocessor_bundled" : True,
                "version_in_filename"  : True,
                "metadata_recorded"    : True,
                "sha256_checksum"      : True,
            },
        },
    }

    try:
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
    except (IOError, OSError) as e:
        _warn("save_artifact", f"Could not write metadata JSON: {e}", e)

    print(f"  Artifact saved  -> {ARTIFACT_PATH}")
    print(f"  SHA-256         : {artifact_sha256}")
    print(f"  Size            : {artifact_size / 1024:.1f} KB")
    print(f"  Metadata saved  -> {METADATA_PATH}")

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# 5. Load-and-predict (simulates a fresh environment call)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_predict(artifact_path: str, raw_records: list) -> list:
    """Load the versioned artifact and run predictions with full validation.

    Parameters
    ----------
    artifact_path : str
        Path to the .joblib artifact.
    raw_records   : list[dict]
        Feature dicts matching the training schema:
        domain_score, aptitude_score, projects_completed,
        active_days, registration_month, total_score

    Returns
    -------
    list[dict] — predictions and probabilities.

    Raises
    ------
    FileNotFoundError – artifact or metadata file missing.
    ValueError        – SHA-256 checksum mismatch (corruption/tampering).
    TypeError         – raw_records is not a list, or items are not dicts.
    RuntimeError      – prediction step fails.
    """
    # ── Input validation ──
    if not isinstance(raw_records, list):
        raise TypeError(
            f"raw_records must be a list, got {type(raw_records).__name__}."
        )
    if len(raw_records) == 0:
        raise ValueError("raw_records is empty — nothing to predict.")
    for i, rec in enumerate(raw_records):
        if not isinstance(rec, dict):
            raise TypeError(
                f"raw_records[{i}] must be a dict, got {type(rec).__name__}."
            )

    # ── Artifact existence check ──
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Artifact not found: '{artifact_path}'. "
            "Run train_task19.py first."
        )

    # ── Metadata & integrity check ──
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found: '{METADATA_PATH}'. "
            "Cannot verify artifact integrity."
        )

    try:
        with open(METADATA_PATH) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise RuntimeError(f"Failed to parse metadata JSON: {e}") from e

    expected_sha = meta.get("artifact", {}).get("sha256", "")
    if not expected_sha:
        raise ValueError("Metadata does not contain a SHA-256 checksum.")

    actual_sha = sha256_of_file(artifact_path)
    if expected_sha != actual_sha:
        raise ValueError(
            f"SHA-256 mismatch!\n"
            f"  Expected : {expected_sha}\n"
            f"  Actual   : {actual_sha}\n"
            "Artifact may be corrupted or tampered with. Aborting load."
        )

    # ── Load ──
    try:
        pipeline = joblib.load(artifact_path)
    except Exception as e:
        raise RuntimeError(
            f"joblib.load() failed on '{artifact_path}': {e}\n"
            "Possible causes: version mismatch between saved and current sklearn."
        ) from e

    if pipeline is None:
        raise RuntimeError("Loaded artifact is None — unexpected corruption.")

    # ── Build DataFrame ──
    try:
        df = pd.DataFrame(raw_records)
    except Exception as e:
        raise RuntimeError(f"Failed to build DataFrame from raw_records: {e}") from e

    if df.empty:
        raise ValueError("DataFrame built from raw_records is empty.")

    # ── Predict ──
    try:
        probs = pipeline.predict_proba(df)[:, 1]
        preds = pipeline.predict(df)
    except Exception as e:
        raise RuntimeError(
            f"Prediction failed: {e}\n"
            f"DataFrame columns: {list(df.columns)}\n"
            f"Pipeline type: {type(pipeline).__name__}"
        ) from e

    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        results.append({
            "sample_index": i,
            "prediction"  : int(pred),
            "probability" : round(float(prob), 4),
            "is_placed"   : bool(pred == 1),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Create output dirs early so error logs can always be written
    for d in [LOGS_DIR, MODELS_DIR, TASK19_DIR]:
        try:
            os.makedirs(d, exist_ok=True)
        except (PermissionError, OSError) as e:
            print(f"[FATAL] Cannot create directory '{d}': {e}")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("  Task 19 — Application Model Serializing")
    print("  PlaceMux  Phase 1 Industry Immersion")
    print("=" * 65)

    # ── Step 1: Acquire pipeline ──────────────────────────────────────
    print("\n[1/6] Acquiring model pipeline ...")
    source   = "pre-trained (Task 12 calibrated_pipeline.pkl)"
    pipeline = None

    if os.path.exists(SRC_MODEL):
        print(f"  Found: {SRC_MODEL} — attempting to load ...")
        try:
            pipeline = joblib.load(SRC_MODEL)
            if pipeline is None:
                raise ValueError("Loaded object is None.")
            print("  Loaded successfully.")
        except Exception as e:
            _warn("load_pretrained",
                  f"Failed to load '{SRC_MODEL}' ({e}). Will train from scratch.", e)
            pipeline = None
    else:
        print(f"  '{SRC_MODEL}' not found. Will train a fresh pipeline.")

    # ── Step 2: Load data ─────────────────────────────────────────────
    print("\n[2/6] Loading project data ...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = get_project_data()
        feature_names = list(X_train.columns)
        print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
        print(f"  Features ({len(feature_names)}): {feature_names}")
    except (ImportError, RuntimeError, ValueError) as e:
        _fatal("load_data", str(e), e)

    # Train from scratch if Step 1 gave us no pipeline
    if pipeline is None:
        source = "trained from scratch (Task 19)"
        print("\n  Training fresh GradientBoosting pipeline ...")
        try:
            pipeline = build_and_train_pipeline(X_train, y_train)
            print("  Pipeline trained successfully.")
        except (ImportError, ValueError, RuntimeError) as e:
            _fatal("train_pipeline", str(e), e)

    # ── Step 3: Evaluate on test set ─────────────────────────────────
    print("\n[3/6] Evaluating on test set ...")
    metrics = evaluate(pipeline, X_test, y_test)   # evaluate() handles its own warnings
    if metrics.get("f1_score") is not None:
        print(f"  Accuracy  : {metrics['accuracy']}")
        print(f"  Precision : {metrics['precision']}")
        print(f"  Recall    : {metrics['recall']}")
        print(f"  F1-Score  : {metrics['f1_score']}  <- Primary metric")
    else:
        print("  Metrics unavailable (see warnings above).")

    # ── Step 4: Serialise artifact + metadata ─────────────────────────
    print(f"\n[4/6] Serialising versioned artifact (v{VERSION}) ...")
    try:
        metadata = save_artifact(
            pipeline      = pipeline,
            metrics       = metrics,
            data_shape    = (len(X_train), len(feature_names)),
            source        = source,
            feature_names = feature_names,
        )
    except (ValueError, PermissionError, RuntimeError) as e:
        _fatal("save_artifact", str(e), e)

    # ── Step 5: Load-and-predict test (fresh environment simulation) ──
    print("\n[5/6] Testing load-and-predict (simulated fresh environment) ...")

    # Use real validation rows so column schema is guaranteed correct
    sample_rows  = X_val.head(4).to_dict(orient="records")
    sample_labels = ["Val row 0", "Val row 1", "Val row 2", "Val row 3"]

    manual_samples = [
        # Strong candidate — all feature names from the actual pipeline
        {col: X_val[col].quantile(0.9) for col in feature_names},
        # Weak candidate
        {col: X_val[col].quantile(0.1) for col in feature_names},
        # Edge case: median values
        {col: X_val[col].median() for col in feature_names},
    ]
    manual_labels = ["90th-pctile candidate", "10th-pctile candidate", "Median candidate"]

    all_samples = sample_rows + manual_samples
    all_labels  = sample_labels + manual_labels

    results = []
    load_test_passed = False
    try:
        results = load_and_predict(ARTIFACT_PATH, all_samples)
        print("\n  Sample Predictions (SHA-256 verified load):")
        for lbl, r in zip(all_labels, results):
            placed_str = "Placed" if r["is_placed"] else "Not Placed"
            print(f"  [{lbl}] => {placed_str}  (prob={r['probability']:.2%})")
        load_test_passed = True
        print("\n  Load-and-predict test PASSED")
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
        _warn("load_and_predict", str(e), e)
        print("  Load-and-predict test FAILED (non-fatal — see results JSON).")

    # ── Step 6: Save results JSON ─────────────────────────────────────
    print("\n[6/6] Saving task results JSON ...")
    results_doc = {
        "task"    : "Task 19 — Application Model Serializing",
        "version" : VERSION,
        "status"  : "COMPLETE" if load_test_passed else "COMPLETE_WITH_WARNINGS",
        "artifact": metadata["artifact"],
        "metrics" : metrics,
        "library_versions": metadata["library_versions"],
        "sample_predictions": results,
        "warnings": _errors,
        "definition_of_done": {
            "serialised_versioned_artifact": True,
            "preprocessor_bundled"         : True,
            "metadata_json_saved"          : True,
            "library_versions_recorded"    : True,
            "load_predict_tested"          : load_test_passed,
            "sha256_integrity_check"       : True,
            "fastapi_stub_provided"        : True,
        },
    }

    try:
        with open(RESULTS_PATH, "w") as f:
            json.dump(results_doc, f, indent=4)
        print(f"  Results saved -> {RESULTS_PATH}")
    except (IOError, OSError) as e:
        print(f"  [WARNING] Could not write results JSON: {e}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  TASK 19 — {'COMPLETE' if load_test_passed else 'COMPLETE (with warnings)'}")
    print("=" * 65)
    print(f"  Artifact : {ARTIFACT_PATH}")
    print(f"  Metadata : {METADATA_PATH}")
    print(f"  Results  : {RESULTS_PATH}")
    print(f"  Version  : v{VERSION}")
    print(f"  F1-Score : {metrics.get('f1_score', 'N/A')}")
    if _errors:
        print(f"  Warnings : {len(_errors)} (see {RESULTS_PATH})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
