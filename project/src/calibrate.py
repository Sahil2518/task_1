"""
Task 12 - Binary Classification: Calibrated, Threshold-Justified Classifier
Provides:
  - get_calibrated_model()         wraps best estimator in CalibratedClassifierCV
  - plot_calibration_curve()       fraction_of_positives vs mean_predicted_value
  - find_cost_optimal_threshold()  cost-matrix search over threshold sweep
  - segment_evaluation()           per-segment stability / fairness check
  - evaluate_calibrated()          full metrics at the chosen threshold
  - cross_fold_stability()         StratifiedKFold fold-by-fold F1 check
"""

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    f1_score,
    fbeta_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.config import CONFIG


# ---------------------------------------------------------------------------
# 1.  Calibration wrapper
# ---------------------------------------------------------------------------

def get_calibrated_model(base_estimator, method="isotonic", cv=5):
    """
    Wrap base_estimator (already a pipeline) with CalibratedClassifierCV.
    cv=5 ensures calibration fold reliability.
    method: 'isotonic' for >=1000 samples, 'sigmoid' for smaller sets.
    """
    return CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=cv,
    )


# ---------------------------------------------------------------------------
# 2.  Calibration curve (reliability diagram)
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    models,
    X_test,
    y_test,
    output_path="logs/calibration_curve.png",
    n_bins=10,
):
    """
    Plot a reliability diagram for one or more models side-by-side.
    A perfectly calibrated model follows the diagonal.
    Also prints Brier scores (lower = better calibration).

    models: dict {name: fitted_model}
    """
    os.makedirs("logs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)

    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
    brier_scores = {}

    for i, (name, model) in enumerate(models.items()):
        probas = model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, probas)
        brier_scores[name] = brier

        CalibrationDisplay.from_predictions(
            y_test,
            probas,
            n_bins=n_bins,
            ax=ax,
            name="{} (Brier={:.4f})".format(name, brier),
            color=colors[i % len(colors)],
        )

    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("  Calibration curve saved -> {}".format(output_path))

    print("\n  Brier Scores (lower is better):")
    for name, score in brier_scores.items():
        print("    {:35s}: {:.4f}".format(name, score))

    return brier_scores


# ---------------------------------------------------------------------------
# 3.  Cost-optimal threshold selection
# ---------------------------------------------------------------------------

def find_cost_optimal_threshold(
    model,
    X_val,
    y_val,
    cost_fp=1.0,
    cost_fn=2.0,
    output_path="logs/threshold_analysis.png",
):
    """
    Sweep thresholds and pick the one minimising:
        total_cost = cost_fp * FP + cost_fn * FN
    Also shows the F1-maximising threshold for comparison.

    Returns: (optimal_threshold, cost_at_threshold, f1_at_threshold)
    """
    probas = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.0, 1.0, 201)

    costs, f1s = [], []
    for t in thresholds:
        preds = (probas >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        costs.append(cost_fp * fp + cost_fn * fn)
        f1s.append(f1_score(y_val, preds, zero_division=0))

    costs = np.array(costs)
    f1s   = np.array(f1s)

    opt_idx    = np.argmin(costs)
    f1_idx     = np.argmax(f1s)
    opt_thresh = float(thresholds[opt_idx])
    f1_thresh  = float(thresholds[f1_idx])

    print("\n  {}".format("=" * 55))
    print("  THRESHOLD SELECTION")
    print("  {}".format("=" * 55))
    print("  Cost weights   : FP={}x  FN={}x".format(cost_fp, cost_fn))
    print("  Cost-optimal   : threshold={:.3f}  (cost={:.1f}, F1={:.4f})".format(
        opt_thresh, costs[opt_idx], f1s[opt_idx]))
    print("  F1-optimal     : threshold={:.3f}  (cost={:.1f}, F1={:.4f})".format(
        f1_thresh, costs[f1_idx], f1s[f1_idx]))
    print("  Operating point chosen: {:.3f}".format(opt_thresh))

    # Plot
    os.makedirs("logs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(thresholds, costs, color="#c44e52", linewidth=2)
    axes[0].axvline(opt_thresh, color="#c44e52", linestyle="--",
                    label="Cost-optimal ({:.3f})".format(opt_thresh))
    axes[0].axvline(f1_thresh, color="#4c72b0", linestyle=":",
                    label="F1-optimal ({:.3f})".format(f1_thresh))
    axes[0].set_xlabel("Threshold", fontsize=11)
    axes[0].set_ylabel("Total Cost (FPx{:.0f} + FNx{:.0f})".format(cost_fp, cost_fn), fontsize=10)
    axes[0].set_title("Cost vs. Threshold", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(thresholds, f1s, color="#55a868", linewidth=2)
    axes[1].axvline(opt_thresh, color="#c44e52", linestyle="--",
                    label="Cost-optimal ({:.3f})".format(opt_thresh))
    axes[1].axvline(f1_thresh, color="#4c72b0", linestyle=":",
                    label="F1-optimal ({:.3f})".format(f1_thresh))
    axes[1].set_xlabel("Threshold", fontsize=11)
    axes[1].set_ylabel("F1-Score", fontsize=11)
    axes[1].set_title("F1 vs. Threshold", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Task 12 - Threshold Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("  Threshold analysis chart saved -> {}".format(output_path))

    return opt_thresh, float(costs[opt_idx]), float(f1s[opt_idx])


# ---------------------------------------------------------------------------
# 4.  Evaluate at chosen threshold
# ---------------------------------------------------------------------------

def evaluate_calibrated(
    model,
    X_test,
    y_test,
    threshold,
    label="Calibrated Model",
    output_path_cm="logs/calibrated_confusion_matrix.png",
):
    """
    Full evaluation at threshold on the sealed test set.
    Returns a dict of metrics and saves a confusion matrix PNG.
    """
    probas = model.predict_proba(X_test)[:, 1]
    preds  = (probas >= threshold).astype(int)

    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, zero_division=0)
    rec   = recall_score(y_test, preds, zero_division=0)
    f1    = f1_score(y_test, preds, zero_division=0)
    auc   = roc_auc_score(y_test, probas)
    brier = brier_score_loss(y_test, probas)

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else None
    fnr_val = float(fn / (fn + tp)) if (fn + tp) > 0 else None

    print("\n  {}".format("=" * 55))
    print("  TEST SET EVALUATION -- {}".format(label))
    print("  Threshold = {:.3f}".format(threshold))
    print("  {}".format("=" * 55))
    print("  Accuracy       : {:.4f}".format(acc))
    print("  Precision      : {:.4f}".format(prec))
    print("  Recall         : {:.4f}".format(rec))
    print("  F1-Score       : {:.4f}".format(f1))
    print("  ROC-AUC        : {:.4f}".format(auc))
    print("  Brier Score    : {:.4f}".format(brier))
    print("  -- Confusion Matrix --")
    print("    TP={}  FP={}  TN={}  FN={}".format(tp, fp, tn, fn))
    print("  Expected error rates:")
    if fpr_val is not None:
        print("    FPR (false alarm) = {:.3f}".format(fpr_val))
    if fnr_val is not None:
        print("    FNR (miss rate)   = {:.3f}".format(fnr_val))

    # Save confusion matrix
    os.makedirs("logs", exist_ok=True)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Placed", "Placed"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (threshold={:.3f})\n{}".format(threshold, label), fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path_cm, dpi=150)
    plt.close()
    print("  Confusion matrix saved -> {}".format(output_path_cm))

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "brier_score": brier,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "FPR": fpr_val,
        "FNR": fnr_val,
    }


# ---------------------------------------------------------------------------
# 5.  Cross-fold stability check
# ---------------------------------------------------------------------------

def cross_fold_stability(
    model,
    X,
    y,
    threshold,
    n_splits=5,
    label="Calibrated Model",
    output_path="logs/fold_stability.png",
):
    """
    StratifiedKFold evaluation: reports per-fold and mean +/- std of F1.
    High std = fragile model. Low std = stable, deployment-ready.
    """
    from sklearn.base import clone
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CONFIG["seed"])

    fold_metrics = []
    print("\n  {}".format("=" * 55))
    print("  CROSS-FOLD STABILITY ({}-fold) -- {}".format(n_splits, label))
    print("  Threshold = {:.3f}".format(threshold))
    print("  {}".format("=" * 55))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_v = y.iloc[train_idx], y.iloc[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        probas = m.predict_proba(X_v)[:, 1]
        preds  = (probas >= threshold).astype(int)

        fold_f1   = f1_score(y_v, preds, zero_division=0)
        fold_prec = precision_score(y_v, preds, zero_division=0)
        fold_rec  = recall_score(y_v, preds, zero_division=0)
        fold_metrics.append({"Fold": fold, "Precision": fold_prec, "Recall": fold_rec, "F1": fold_f1})
        print("    Fold {}: Prec={:.4f}  Rec={:.4f}  F1={:.4f}".format(
            fold, fold_prec, fold_rec, fold_f1))

    df = pd.DataFrame(fold_metrics)
    mean_f1 = df["F1"].mean()
    std_f1  = df["F1"].std()
    print("\n  Mean F1 = {:.4f} +/- {:.4f}".format(mean_f1, std_f1))
    if std_f1 < 0.03:
        print("  Verdict: STABLE -- std < 0.03 across folds.")
    else:
        print("  Verdict: UNSTABLE -- consider more data or regularisation.")

    # Plot fold bars
    os.makedirs("logs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df["Fold"], df["F1"], color="#4c72b0", edgecolor="white")
    ax.bar_label(bars, fmt="{:.4f}", padding=3, fontsize=9)
    ax.axhline(mean_f1, color="#c44e52", linestyle="--", linewidth=1.5,
               label="Mean F1 = {:.4f}".format(mean_f1))
    ax.fill_between(
        [0.5, n_splits + 0.5],
        mean_f1 - std_f1, mean_f1 + std_f1,
        alpha=0.15, color="#c44e52",
        label="+/-1 std ({:.4f})".format(std_f1),
    )
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("Cross-Fold Stability ({}-fold StratifiedKFold)\n{}".format(n_splits, label),
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("  Fold stability chart saved -> {}".format(output_path))

    return df, mean_f1, std_f1


# ---------------------------------------------------------------------------
# 6.  Segment fairness / stability check
# ---------------------------------------------------------------------------

def segment_evaluation(
    model,
    X_test,
    y_test,
    threshold,
    segment_col="projects_completed",
    output_path="logs/segment_evaluation.png",
):
    """
    Split the test set on segment_col and evaluate F1 per segment.
    Surfaces hidden per-group performance gaps (fairness / stability detector).
    """
    probas = model.predict_proba(X_test)[:, 1]
    preds  = (probas >= threshold).astype(int)

    results_df = X_test.copy()
    results_df["y_true"] = np.array(y_test)
    results_df["y_pred"] = preds
    results_df["prob"]   = probas

    # Bin continuous segment columns
    seg = results_df[segment_col]
    if seg.nunique() > 8:
        results_df["segment"] = pd.qcut(seg, q=4, duplicates="drop").astype(str)
    else:
        results_df["segment"] = seg.astype(str)

    rows = []
    print("\n  {}".format("=" * 55))
    print("  SEGMENT EVALUATION -- by '{}'".format(segment_col))
    print("  Threshold = {:.3f}".format(threshold))
    print("  {}".format("=" * 55))

    for seg_val, grp in results_df.groupby("segment"):
        if len(grp) < 5:
            continue
        f1_s  = f1_score(grp["y_true"], grp["y_pred"], zero_division=0)
        prec  = precision_score(grp["y_true"], grp["y_pred"], zero_division=0)
        rec   = recall_score(grp["y_true"], grp["y_pred"], zero_division=0)
        size  = len(grp)
        rows.append({"Segment": seg_val, "N": size, "Precision": prec, "Recall": rec, "F1": f1_s})
        print("    {:20s}  N={:4d}  Prec={:.4f}  Rec={:.4f}  F1={:.4f}".format(
            str(seg_val), size, prec, rec, f1_s))

    seg_df = pd.DataFrame(rows)

    # Plot
    os.makedirs("logs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(seg_df))
    width = 0.28
    ax.bar(x - width, seg_df["Precision"], width, label="Precision", color="#4c72b0")
    ax.bar(x,          seg_df["F1"],        width, label="F1",        color="#dd8452")
    ax.bar(x + width,  seg_df["Recall"],    width, label="Recall",    color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(seg_df["Segment"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Segment Evaluation -- by '{}' (threshold={:.3f})".format(segment_col, threshold),
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("  Segment evaluation chart saved -> {}".format(output_path))

    return seg_df
