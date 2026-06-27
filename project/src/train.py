from src.data import load_data
from src.model import get_baseline, get_model
from src.evaluate import (
    evaluate_model,
    compare_baseline,
    inspect_errors,
    log_results,
    plot_confusion_matrix,
    evaluate_with_threshold,
    plot_curves
)

from src.preprocess import (
    get_feature_types,
    get_preprocessor
)

import joblib
import os


def main():

    # ── Step 1: Load Data (train / val / test) ──
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Step 2: Preprocessing ──
    numeric_features, categorical_features = get_feature_types(
        X_train
    )

    preprocessor = get_preprocessor(
        numeric_features,
        categorical_features
    )

    # Save feature names before transform (for error analysis)
    feature_names = list(X_train.columns)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    from sklearn.pipeline import Pipeline
    import json

    # ── Step 3: Baseline (majority-class) ──
    baseline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', get_baseline())
    ])
    baseline.fit(X_train, y_train)
    baseline_acc, _ = evaluate_model(
        baseline, X_val, y_val,
        label="BASELINE (DummyClassifier — majority class)"
    )

    # ── Step 4: First Real Model (Logistic Regression Pipeline) ──
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', get_model())
    ])
    model.fit(X_train, y_train)
    
    # Base evaluation
    model_acc, _ = evaluate_model(
        model, X_val, y_val,
        label="FIRST MODEL (LogisticRegression)"
    )
    plot_confusion_matrix(
        model,
        X_val,
        y_val
    )
    
    # Plot ROC and PR curves
    plot_curves(model, X_val, y_val)
    
    # ── Step 4.5: Threshold selection ──
    custom_threshold = 0.4
    
    acc_t, prec_t, rec_t, f1_t, preds_t = evaluate_with_threshold(
        model, X_val, y_val, threshold=custom_threshold, 
        label="LOGISTIC REGRESSION (Custom Threshold)"
    )

    # Save Unified Pipeline Artifact
    joblib.dump(
        model,
        "models/pipeline.pkl"
    )

    # Save Metrics Artifact
    metrics = {
        "baseline_accuracy": float(baseline_acc),
        "model_accuracy": float(acc_t),
        "precision": float(prec_t),
        "recall": float(rec_t),
        "f1": float(f1_t),
        "threshold": custom_threshold
    }
    with open("logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("  Metrics saved to logs/metrics.json")

    # ── Step 5: Compare against baseline ──
    compare_baseline(baseline_acc, acc_t)

    # ── Step 6: Inspect worst errors (using custom threshold) ──
    # Re-assign predictions based on custom threshold for error analysis
    model.predict = lambda X: (model.predict_proba(X)[:, 1] >= custom_threshold).astype(int)
    
    inspect_errors(
        model, X_val, y_val,
        feature_names=feature_names
    )

    # ── Step 7: Log results (CSV) ──
    error_notes = f"binary classification, threshold={custom_threshold}, single-pipeline"
    log_results(baseline_acc, acc_t, "LogisticRegression", error_notes, prec_t, rec_t, f1_t)

    print("\n  Next improvement: Tune hyperparameters or test on holdout set.")

if __name__ == "__main__":
    main()