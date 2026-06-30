from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def evaluate_model(model, X_val, y_val, label="Model"):
    """Evaluate a single model on validation data."""
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_val, predictions,
        target_names=["Class 0", "Class 1"]
    ))

    return accuracy, predictions

def plot_confusion_matrix(model, X_val, y_val):
    """Plot confusion matrix for the validation set."""

    # Step 1: Predict the validation data
    predictions = model.predict(X_val)

    # Step 2: Compute confusion matrix
    cm = confusion_matrix(
        y_val,
        predictions
    )

    # Step 3: Create display object
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    # Step 4: Draw the matrix
    disp.plot(cmap="Blues")

    # Step 5: Add title
    plt.title("Confusion Matrix")

    # Step 6: Save graph
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/confusion_matrix.png")
    plt.close()
    print("  Confusion matrix saved to logs/confusion_matrix.png")

def compare_baseline(baseline_acc, model_acc):
    """Print whether the real model actually beats the baseline."""
    diff = model_acc - baseline_acc

    print(f"\n{'='*50}")
    print(f"  BASELINE COMPARISON")
    print(f"{'='*50}")
    print(f"  Baseline Accuracy : {baseline_acc:.4f}")
    print(f"  Model Accuracy    : {model_acc:.4f}")
    print(f"  Improvement       : {diff:+.4f}")

    if diff > 0:
        print(f"  Verdict: Model BEATS baseline by {diff:.4f}")
    elif diff == 0:
        print(f"  Verdict: Model TIES with baseline (no improvement)")
    else:
        print(f"  Verdict: Model is WORSE than baseline!")

    print()
    return diff


def inspect_errors(model, X_val, y_val, feature_names):
    """Inspect the worst errors to find patterns."""
    predictions = model.predict(X_val)
    errors = np.array(predictions) != np.array(y_val)

    if not np.any(errors):
        print("  No errors on validation set!")
        return

    class_names = ["Class 0", "Class 1"]

    X_val_df = pd.DataFrame(X_val, columns=feature_names) \
        if not isinstance(X_val, pd.DataFrame) \
        else X_val.copy()

    X_val_df = X_val_df.reset_index(drop=True)
    y_val_reset = np.array(y_val)

    error_df = X_val_df[errors].copy()
    error_df["true_label"] = [class_names[i] for i in y_val_reset[errors]]
    error_df["predicted"] = [class_names[i] for i in predictions[errors]]

    print(f"\n{'='*50}")
    print(f"  ERROR ANALYSIS")
    print(f"{'='*50}")
    print(f"  Total errors: {errors.sum()} / {len(y_val)}")
    print(f"\n  Misclassified samples:")
    print(error_df.to_string(index=False))

    # Pattern: which classes get confused?
    print(f"\n  Confusion patterns:")
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i != j:
                count = ((y_val_reset == i) & (predictions == j)).sum()
                if count > 0:
                    print(f"    {true_name} misclassified as {pred_name}: {count}")

    return error_df

from sklearn.metrics import precision_score, recall_score, f1_score, RocCurveDisplay, PrecisionRecallDisplay

def evaluate_with_threshold(model, X_val, y_val, threshold=0.5, label="Model with Custom Threshold"):
    """Evaluate model with a custom decision threshold."""
    # Get probabilities for the positive class (class 1)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_val)[:, 1]
    else:
        # Fallback if model doesn't support predict_proba
        probas = model.decision_function(X_val)
        
    predictions = (probas >= threshold).astype(int)
    
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {label} (Threshold: {threshold})")
    print(f"{'='*50}")
    print(f"  Validation Accuracy : {accuracy:.4f}")
    print(f"  Precision           : {precision:.4f}")
    print(f"  Recall              : {recall:.4f}")
    print(f"  F1-Score            : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_val, predictions,
        target_names=["Class 0", "Class 1"]
    ))

    # Save confusion matrix for this threshold
    cm = confusion_matrix(y_val, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Threshold={threshold})")
    os.makedirs("logs", exist_ok=True)
    plt.savefig(f"logs/confusion_matrix_t{threshold}.png")
    plt.close()
    
    return accuracy, precision, recall, f1, predictions

def plot_curves(model, X_val, y_val):
    """Plot ROC and Precision-Recall curves."""
    os.makedirs("logs", exist_ok=True)
    
    # ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_val, y_val, ax=ax)
    plt.title("ROC Curve")
    plt.savefig("logs/roc_curve.png")
    plt.close()
    
    # Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_val, y_val, ax=ax)
    plt.title("Precision-Recall Curve")
    plt.savefig("logs/pr_curve.png")
    plt.close()
    print("  ROC and PR curves saved to logs/")


def log_results(baseline_acc, model_acc, model_name, notes="", precision=None, recall=None, f1=None):
    """Append results to the experiment log."""
    os.makedirs("logs", exist_ok=True)

    results = pd.DataFrame([
        {
            "model": "DummyClassifier (baseline)",
            "accuracy": baseline_acc,
            "precision": None,
            "recall": None,
            "f1": None,
            "notes": "majority-class baseline"
        },
        {
            "model": model_name,
            "accuracy": model_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "notes": notes
        }
    ])

    results.to_csv("logs/results.csv", index=False)
    print(f"  Results saved to logs/results.csv")

    return results

from sklearn.inspection import PartialDependenceDisplay

def plot_partial_dependence(model, X_val, features, feature_names):
    """Plot Partial Dependence for the top features."""
    os.makedirs("logs", exist_ok=True)
    
    # Convert X_val to DataFrame if it's not already, for better feature name handling
    X_val_df = pd.DataFrame(X_val, columns=feature_names) if not isinstance(X_val, pd.DataFrame) else X_val.copy()
    
    print(f"\n  Generating Partial Dependence Plots for top features: {features}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    disp = PartialDependenceDisplay.from_estimator(
        model,
        X_val_df,
        features=features,
        ax=ax,
        grid_resolution=50
    )
    plt.suptitle("Partial Dependence Plots (PDP)")
    plt.tight_layout()
    plt.savefig("logs/pdp_plot.png")
    plt.close()
    print("  Partial Dependence Plots saved to logs/pdp_plot.png")