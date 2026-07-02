"""
Task 12 - Binary Classification: Calibrated, Threshold-Justified Classifier
Main training script.  Follows the same pattern as Task 11 (train_ensemble.py).

Steps:
  1. Load data (same deterministic pipeline as Tasks 8-11)
  2. Train the best single model as the base (Gradient Boosting, from Task 10/11)
  3. Calibrate it with CalibratedClassifierCV (isotonic, cv=5)
  4. Verify calibration with a reliability diagram + Brier score
  5. Pick the cost-optimal threshold on the validation set
  6. Evaluate on the sealed test set at that threshold
  7. Cross-fold stability check (StratifiedKFold)
  8. Segment evaluation (per key feature group)
  9. Document the operating point and expected error rates in JSON
 10. Save the calibrated pipeline (production artifact)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data import load_data
from src.preprocess import get_feature_types, get_preprocessor
from src.ensemble import get_gradient_boosting          # reuse Task 11 base
from src.calibrate import (
    get_calibrated_model,
    plot_calibration_curve,
    find_cost_optimal_threshold,
    evaluate_calibrated,
    cross_fold_stability,
    segment_evaluation,
)

COST_FP = 1.0   # Cost of false positive (predict placed but actually not)
COST_FN = 2.0   # Cost of false negative (miss a real placement -- worse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    print("\n" + "=" * 65)
    print("  Task 12 -- Calibrated Binary Classification")
    print("  PlaceMux  Phase 1 Industry Immersion")
    print("=" * 65)

    # STEP 1: Load Data
    print("\n[1/9] Loading data ...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print("  Train: {}  Val: {}  Test: {}".format(len(X_train), len(X_val), len(X_test)))

    # Combine train+val for fitting the calibrator (test stays sealed)
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    # STEP 2: Build & Fit Base Pipeline
    print("\n[2/9] Training base Gradient Boosting pipeline ...")
    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    base_pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf",  get_gradient_boosting()),
    ])
    base_pipeline.fit(X_train, y_train)   # fit on train only (val used for calibration)

    base_preds = base_pipeline.predict(X_test)
    base_f1    = f1_score(y_test, base_preds, zero_division=0)
    print("  Base GB -- Test F1: {:.4f}".format(base_f1))

    # STEP 3: Calibrate
    print("\n[3/9] Calibrating with CalibratedClassifierCV (isotonic, cv=5) ...")
    # Refit on the full trainval set inside CalibratedClassifierCV
    numeric_features2, categorical_features2 = get_feature_types(X_trainval)
    preprocessor2 = get_preprocessor(numeric_features2, categorical_features2)
    base_for_cal = Pipeline([
        ("prep", preprocessor2),
        ("clf",  get_gradient_boosting()),
    ])
    calibrated_model = get_calibrated_model(base_for_cal, method="isotonic", cv=5)
    calibrated_model.fit(X_trainval, y_trainval)

    cal_preds = calibrated_model.predict(X_test)
    cal_f1    = f1_score(y_test, cal_preds, zero_division=0)
    print("  Calibrated -- Test F1 (default t=0.5): {:.4f}".format(cal_f1))

    # STEP 4: Calibration Curve
    print("\n[4/9] Plotting calibration curve (reliability diagram) ...")
    brier_scores = plot_calibration_curve(
        models={"Base GB (uncalibrated)": base_pipeline, "Calibrated GB": calibrated_model},
        X_test=X_test,
        y_test=y_test,
        output_path="logs/calibration_curve.png",
        n_bins=10,
    )

    # STEP 5: Cost-Optimal Threshold
    print("\n[5/9] Finding cost-optimal threshold (on validation set) ...")
    opt_threshold, opt_cost, opt_f1 = find_cost_optimal_threshold(
        model=calibrated_model,
        X_val=X_val,
        y_val=y_val,
        cost_fp=COST_FP,
        cost_fn=COST_FN,
        output_path="logs/threshold_analysis.png",
    )

    # STEP 6: Test-Set Evaluation at Chosen Threshold
    print("\n[6/9] Evaluating on sealed test set (threshold={:.3f}) ...".format(opt_threshold))
    test_metrics = evaluate_calibrated(
        model=calibrated_model,
        X_test=X_test,
        y_test=y_test,
        threshold=opt_threshold,
        label="Calibrated GB (Task 12)",
        output_path_cm="logs/calibrated_confusion_matrix.png",
    )

    # STEP 7: Cross-Fold Stability
    print("\n[7/9] Cross-fold stability check (5-fold StratifiedKFold) ...")
    numeric_features3, categorical_features3 = get_feature_types(X_trainval)
    preprocessor3 = get_preprocessor(numeric_features3, categorical_features3)
    base_for_cv = Pipeline([
        ("prep", preprocessor3),
        ("clf",  get_gradient_boosting()),
    ])
    cal_for_cv = get_calibrated_model(base_for_cv, method="isotonic", cv=5)

    fold_df, mean_f1, std_f1 = cross_fold_stability(
        model=cal_for_cv,
        X=X_trainval,
        y=y_trainval,
        threshold=opt_threshold,
        n_splits=5,
        label="Calibrated GB (Task 12)",
        output_path="logs/fold_stability.png",
    )

    # STEP 8: Segment Evaluation
    print("\n[8/9] Per-segment evaluation ...")
    seg_df = segment_evaluation(
        model=calibrated_model,
        X_test=X_test,
        y_test=y_test,
        threshold=opt_threshold,
        segment_col="projects_completed",
        output_path="logs/segment_evaluation.png",
    )

    # STEP 9: Save Model & Document Operating Point
    print("\n[9/9] Saving calibrated model and metrics ...")

    joblib.dump(calibrated_model, "models/calibrated_pipeline.pkl")
    print("  Calibrated model saved -> models/calibrated_pipeline.pkl")

    # Document the operating point
    operating_point = {
        "task": "Task 12 -- Binary Classification (Calibrated)",
        "model": "CalibratedClassifierCV(GradientBoosting, isotonic, cv=5)",
        "calibration_method": "isotonic regression",
        "cost_weights": {"FP": COST_FP, "FN": COST_FN},
        "operating_threshold": opt_threshold,
        "brier_score_uncalibrated": brier_scores.get("Base GB (uncalibrated)"),
        "brier_score_calibrated":   brier_scores.get("Calibrated GB"),
        "test_metrics": test_metrics,
        "cross_fold_stability": {
            "n_folds": 5,
            "mean_f1": float(mean_f1),
            "std_f1":  float(std_f1),
            "stable":  bool(std_f1 < 0.03),
            "folds": fold_df.to_dict(orient="records"),
        },
        "segment_evaluation": seg_df.to_dict(orient="records"),
        "expected_error_rates": {
            "FPR (false alarm rate)": test_metrics.get("FPR"),
            "FNR (miss rate)":        test_metrics.get("FNR"),
        },
        "pitfalls_avoided": {
            "uncalibrated_probabilities":
                "CalibratedClassifierCV with isotonic regression applied",
            "hidden_segment_failure":
                "Per-segment F1 checked across projects_completed groups",
            "no_documented_operating_point":
                "Threshold={:.3f}, cost-weighted selection documented".format(opt_threshold),
        },
    }

    with open("logs/task12_operating_point.json", "w") as f:
        json.dump(operating_point, f, indent=4)
    print("  Operating point documented -> logs/task12_operating_point.json")

    # Summary
    print("\n" + "=" * 65)
    print("  TASK 12 -- SUMMARY")
    print("=" * 65)
    print("  Calibration:   isotonic, cv=5  | Brier {:.4f} -> {:.4f}".format(
        brier_scores.get("Base GB (uncalibrated)", 0),
        brier_scores.get("Calibrated GB", 0)))
    print("  Operating pt:  threshold = {:.3f}  (cost: FPx{} | FNx{})".format(
        opt_threshold, COST_FP, COST_FN))
    print("  Test F1:       {:.4f}   ROC-AUC: {:.4f}".format(
        test_metrics["f1"], test_metrics["roc_auc"]))
    print("  Stability:     Mean F1 = {:.4f} +/- {:.4f}  ({})".format(
        mean_f1, std_f1, "STABLE" if std_f1 < 0.03 else "UNSTABLE"))
    fpr_v = test_metrics.get("FPR") or 0
    fnr_v = test_metrics.get("FNR") or 0
    print("  FPR / FNR:     {:.3f} / {:.3f}".format(fpr_v, fnr_v))
    print("  Model file:    models/calibrated_pipeline.pkl")
    print("=" * 65)
    print("\n  Task 12 Complete -- Calibrated classifier packaged for serving.\n")


if __name__ == "__main__":
    main()
