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


# ══════════════════════════════════════════════════════════════════
# Task 11 — Ensemble-specific evaluation helpers
# ══════════════════════════════════════════════════════════════════

def compare_models_table(results: dict, set_name: str = "Test"):
    """
    Pretty-print a table of {model_name: (acc, prec, rec, f1)} and
    return a sorted DataFrame. 'results' keys are model names.
    """
    rows = []
    for name, (acc, prec, rec, f1) in results.items():
        rows.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    print(f"\n{'='*65}")
    print(f"  MODEL COMPARISON — {set_name} Set")
    print(f"{'='*65}")
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print(f"{'='*65}\n")
    return df


def diversity_matrix(named_predictions: dict, X_test, y_test):
    """
    Compute pairwise prediction *disagreement* between models.
    High disagreement = diverse ensemble = better error averaging.
    named_predictions: {name: np.array of binary predictions}
    """
    names = list(named_predictions.keys())
    preds = {n: np.asarray(p) for n, p in named_predictions.items()}

    print(f"\n{'='*55}")
    print(f"  DIVERSITY MATRIX (pairwise disagreement rate)")
    print(f"  (Higher = models disagree more = better diversity)")
    print(f"{'='*55}")

    matrix = {}
    for i, a in enumerate(names):
        row = {}
        for j, b in enumerate(names):
            if a == b:
                row[b] = 0.0
            else:
                disagree = np.mean(preds[a] != preds[b])
                row[b] = disagree
        matrix[a] = row

    df = pd.DataFrame(matrix)
    print(df.to_string(float_format="{:.3f}".format))
    print()
    return df


def plot_ensemble_comparison(results: dict, output_path: str = "logs/ensemble_comparison.png"):
    """
    Bar chart comparing F1 scores of all models.
    Ensembles are highlighted in a distinct colour.
    """
    import matplotlib.pyplot as plt
    os.makedirs("logs", exist_ok=True)

    names = list(results.keys())
    f1_scores = [v[3] for v in results.values()]   # (acc, prec, rec, f1)
    colors = [
        "#e07b54" if "Voting" in n or "Stacking" in n else "#5b8db8"
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, f1_scores, color=colors, edgecolor="white", height=0.55)
    ax.bar_label(bars, fmt="{:.4f}", padding=4, fontsize=9)
    ax.set_xlabel("F1-Score", fontsize=11)
    ax.set_title("Task 11 — Ensemble vs. Single-Model F1 Comparison", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e07b54", label="Ensemble"),
        Patch(facecolor="#5b8db8", label="Single model"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Ensemble comparison chart saved to {output_path}")


def evaluate_ensemble_lift(single_results: dict, ensemble_results: dict):
    """
    Print the lift that the best ensemble achieves over the best single model.
    single_results / ensemble_results: {name: (acc, prec, rec, f1)}
    """
    best_single_name = max(single_results, key=lambda k: single_results[k][3])
    best_single_f1   = single_results[best_single_name][3]

    best_ens_name = max(ensemble_results, key=lambda k: ensemble_results[k][3])
    best_ens_f1   = ensemble_results[best_ens_name][3]

    lift = best_ens_f1 - best_single_f1

    print(f"\n{'='*55}")
    print(f"  ENSEMBLE LIFT SUMMARY")
    print(f"{'='*55}")
    print(f"  Best single model : {best_single_name}  (F1={best_single_f1:.4f})")
    print(f"  Best ensemble     : {best_ens_name}  (F1={best_ens_f1:.4f})")
    print(f"  Lift              : {lift:+.4f}")
    if lift > 0:
        print(f"  Verdict: Ensemble BEATS best single model ✅")
    elif lift == 0:
        print(f"  Verdict: Ensemble TIES best single model (no lift)")
    else:
        print(f"  Verdict: Ensemble is WORSE than best single model ❌")
    print(f"{'='*55}\n")
    return lift