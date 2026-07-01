"""
Task 11 — Ensemble Learning
Train 3 diverse base models + Voting & Stacking ensembles, then
compare everything on the held-out test set and document the lift.
"""

import os
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data import load_data
from src.preprocess import get_feature_types, get_preprocessor
from src.ensemble import (
    get_logistic_regression,
    get_random_forest,
    get_gradient_boosting,
    get_voting_ensemble,
    get_stacking_ensemble,
)
from src.evaluate import (
    compare_models_table,
    diversity_matrix,
    plot_ensemble_comparison,
    evaluate_ensemble_lift,
)


# ─────────────────────────────────────────────
# Helper: fit a full pipeline and score it
# ─────────────────────────────────────────────

def fit_single(name, clf, preprocessor, X_train, y_train, X_test, y_test):
    """Wrap clf in a pipeline, fit on train, evaluate on test."""
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    print(f"  Training {name} ...")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    print(f"    {name}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    return pipe, preds, (acc, prec, rec, f1)


def fit_ensemble(name, ensemble, X_train, y_train, X_test, y_test):
    """Fit a pre-built ensemble (already has preprocessor inside each leg)."""
    print(f"  Training {name} ...")
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    print(f"    {name}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    return ensemble, preds, (acc, prec, rec, f1)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    # ── 1. Load Data ──────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Combine train + val for final model fitting (test stays sealed)
    X_trainval = X_train._append(X_val, ignore_index=True)  # pandas ≥2 uses concat
    try:
        import pandas as pd
        X_trainval = pd.concat([X_train, X_val], ignore_index=True)
        y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    except Exception:
        X_trainval = X_train
        y_trainval = y_train

    # ── 2. Preprocessor (shared template — cloned per sub-pipeline) ───
    numeric_features, categorical_features = get_feature_types(X_train)

    # We call get_preprocessor() fresh each time so each leg gets its own fitted copy
    def fresh_preprocessor():
        return get_preprocessor(numeric_features, categorical_features)

    # ── 3. Train Base Models ──────────────────────────────────────
    print("\n" + "="*55)
    print("  STEP 1 — Training Diverse Base Models")
    print("="*55)

    lr_pipe,  lr_preds,  lr_scores  = fit_single("Logistic Regression", get_logistic_regression(), fresh_preprocessor(), X_trainval, y_trainval, X_test, y_test)
    rf_pipe,  rf_preds,  rf_scores  = fit_single("Random Forest",       get_random_forest(),       fresh_preprocessor(), X_trainval, y_trainval, X_test, y_test)
    gb_pipe,  gb_preds,  gb_scores  = fit_single("Gradient Boosting",   get_gradient_boosting(),   fresh_preprocessor(), X_trainval, y_trainval, X_test, y_test)

    single_results = {
        "Logistic Regression": lr_scores,
        "Random Forest":       rf_scores,
        "Gradient Boosting":   gb_scores,
    }
    single_preds = {
        "Logistic Regression": lr_preds,
        "Random Forest":       rf_preds,
        "Gradient Boosting":   gb_preds,
    }

    # ── 4. Train Ensembles ────────────────────────────────────────
    print("\n" + "="*55)
    print("  STEP 2 — Training Ensemble Models")
    print("="*55)

    voting_ens   = get_voting_ensemble(fresh_preprocessor())
    stacking_ens = get_stacking_ensemble(fresh_preprocessor())

    voting_model,   voting_preds,   voting_scores   = fit_ensemble("Voting Ensemble (soft)",   voting_ens,   X_trainval, y_trainval, X_test, y_test)
    stacking_model, stacking_preds, stacking_scores = fit_ensemble("Stacking Ensemble (OOF5)", stacking_ens, X_trainval, y_trainval, X_test, y_test)

    ensemble_results = {
        "Voting Ensemble (soft)":   voting_scores,
        "Stacking Ensemble (OOF5)": stacking_scores,
    }
    ensemble_preds = {
        "Voting Ensemble (soft)":   voting_preds,
        "Stacking Ensemble (OOF5)": stacking_preds,
    }

    # ── 5. Compare All Models ─────────────────────────────────────
    print("\n" + "="*55)
    print("  STEP 3 — Model Comparison (Test Set)")
    print("="*55)

    all_results = {**single_results, **ensemble_results}
    comparison_df = compare_models_table(all_results, set_name="Test")

    # ── 6. Diversity Check ────────────────────────────────────────
    print("\n" + "="*55)
    print("  STEP 4 — Diversity Check")
    print("="*55)

    all_preds = {**single_preds, **ensemble_preds}
    div_df = diversity_matrix(all_preds, X_test, y_test)

    # ── 7. Plot Comparison Chart ──────────────────────────────────
    print("\n" + "="*55)
    print("  STEP 5 — Generating Charts")
    print("="*55)

    plot_ensemble_comparison(all_results)

    # ── 8. Lift Summary ───────────────────────────────────────────
    print("\n" + "="*55)
    print("  STEP 6 — Lift Analysis")
    print("="*55)

    lift = evaluate_ensemble_lift(single_results, ensemble_results)

    # ── 9. Save Best Ensemble ──────────────────────────────────────
    best_ens_name = max(ensemble_results, key=lambda k: ensemble_results[k][3])
    best_ens_model = voting_model if "Voting" in best_ens_name else stacking_model

    joblib.dump(best_ens_model, "models/ensemble_pipeline.pkl")
    print(f"  Best ensemble ({best_ens_name}) saved to models/ensemble_pipeline.pkl")

    # Also save individual models for inference options
    joblib.dump(lr_pipe,  "models/lr_pipeline.pkl")
    joblib.dump(rf_pipe,  "models/rf_pipeline.pkl")
    joblib.dump(gb_pipe,  "models/gb_pipeline.pkl")
    print("  Individual model pipelines saved to models/")

    # ── 10. Save Metrics JSON ─────────────────────────────────────
    def scores_to_dict(s):
        return {"accuracy": float(s[0]), "precision": float(s[1]),
                "recall": float(s[2]), "f1": float(s[3])}

    metrics = {
        "task": "Task 11 — Ensemble Learning",
        "best_single_model": max(single_results,  key=lambda k: single_results[k][3]),
        "best_ensemble":     max(ensemble_results, key=lambda k: ensemble_results[k][3]),
        "lift_f1": float(lift),
        "single_models": {k: scores_to_dict(v) for k, v in single_results.items()},
        "ensembles":     {k: scores_to_dict(v) for k, v in ensemble_results.items()},
    }
    with open("logs/ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("  Metrics saved to logs/ensemble_metrics.json")

    # ── 11. Save Comparison CSV ───────────────────────────────────
    comparison_df.to_csv("logs/ensemble_comparison.csv", index=False)
    print("  Comparison table saved to logs/ensemble_comparison.csv")

    print("\n  Task 11 Complete — Ensemble Training & Evaluation Done.\n")


if __name__ == "__main__":
    main()
