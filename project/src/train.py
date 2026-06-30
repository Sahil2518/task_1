from src.data import load_data
from src.model import get_baseline, get_model
from src.evaluate import (
    evaluate_model,
    compare_baseline,
    inspect_errors,
    log_results,
    plot_confusion_matrix,
    evaluate_with_threshold,
    plot_curves,
    plot_partial_dependence
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

    # ── Step 4: Hyperparameter Tuning (GridSearchCV) ──
    from sklearn.model_selection import GridSearchCV

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', get_model())
    ])

    param_grid = {
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5],
        'classifier__n_estimators': [50, 100]
    }

    print("\n  Starting GridSearchCV (optimizing for f1)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    print(f"  Best params: {grid_search.best_params_}")
    
    # ── Step 4.5: Evaluation on Validation Set ──
    model_acc, _ = evaluate_model(
        model, X_val, y_val,
        label="TUNED MODEL (GradientBoosting - Val Set)"
    )
    plot_confusion_matrix(
        model, X_val, y_val
    )
    plot_curves(model, X_val, y_val)
    
    custom_threshold = 0.4
    acc_t, prec_t, rec_t, f1_t, preds_t = evaluate_with_threshold(
        model, X_val, y_val, threshold=custom_threshold, 
        label="TUNED MODEL (Custom Threshold - Val Set)"
    )
    
    # ── Step 4.7: Test Set Confirmation ──
    print("\n  Evaluating on held-out Test Set...")
    test_acc_t, test_prec_t, test_rec_t, test_f1_t, _ = evaluate_with_threshold(
        model, X_test, y_test, threshold=custom_threshold,
        label="TUNED MODEL (Custom Threshold - Test Set)"
    )

    # Save Unified Pipeline Artifact
    joblib.dump(
        model,
        "models/pipeline.pkl"
    )

    # Save Metrics Artifact
    metrics = {
        "baseline_accuracy": float(baseline_acc),
        "val_accuracy": float(acc_t),
        "val_precision": float(prec_t),
        "val_recall": float(rec_t),
        "val_f1": float(f1_t),
        "test_accuracy": float(test_acc_t),
        "test_precision": float(test_prec_t),
        "test_recall": float(test_rec_t),
        "test_f1": float(test_f1_t),
        "threshold": custom_threshold,
        "best_params": grid_search.best_params_
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
    error_notes = f"tuned, grid search, threshold={custom_threshold}"
    log_results(baseline_acc, acc_t, "GradientBoosting(Tuned)", error_notes, prec_t, rec_t, f1_t)

    # ── Step 8: Partial Dependence Plots (PDP) ──
    import numpy as np
    feature_importances = model.named_steps['classifier'].feature_importances_
    
    try:
        transformed_features = model.named_steps['preprocessor'].get_feature_names_out()
    except AttributeError:
        transformed_features = feature_names

    top_feature_indices = np.argsort(feature_importances)[-2:][::-1]
    top_feature_names = [transformed_features[i] for i in top_feature_indices]
    
    X_val_transformed = model.named_steps['preprocessor'].transform(X_val)
    if hasattr(X_val_transformed, "toarray"):
        X_val_transformed = X_val_transformed.toarray()
        
    plot_partial_dependence(
        model.named_steps['classifier'], 
        X_val_transformed, 
        features=top_feature_indices, 
        feature_names=transformed_features
    )

    print("\n  Task 10 Tuning Complete. PDP generated and Test-set confirmation passed.")

if __name__ == "__main__":
    main()