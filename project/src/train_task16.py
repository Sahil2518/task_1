"""
Task 16 — Model Validation & K-Fold
PlaceMux Phase 1 Industry Immersion

Steps:
  1. Choose StratifiedKFold (classification on imbalanced data).
  2. Compare 3 candidate models (LR, RF, GB) on the SAME folds.
  3. Report per-fold scores, mean, and std – never just the best fold.
  4. Nested CV for GB (inner loop tunes max_depth; outer loop evaluates).
  5. Conclude which model generalises best.
  6. Save JSON report + plots.

Error-handling:  every major step is wrapped in try/except with
graceful fallback so the script never silently swallows a failure.
"""

import os
import sys
import json
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_score,
    GridSearchCV,
    cross_validate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.dummy import DummyClassifier

# ── project imports ────────────────────────────────────────────────────────────
try:
    from src.data import load_data
    from src.preprocess import get_feature_types, get_preprocessor
    from src.config import CONFIG
except ImportError as _ie:
    print(f"[FATAL] Could not import project modules: {_ie}")
    print("        Run this script from the project root directory.")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────
SEED        = 42
N_SPLITS    = 5
SCORING     = "f1"
LOGS_DIR    = "logs"
PLOTS_DIR   = os.path.join(LOGS_DIR, "task16_plots")
RESULTS_JSON = os.path.join(LOGS_DIR, "task16_kfold_results.json")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_json(obj):
    """Recursively convert numpy types so json.dump doesn't choke."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return obj


def build_pipeline(clf, preprocessor):
    """Wrap a classifier in a full sklearn Pipeline with the preprocessor."""
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def get_candidate_models(preprocessor):
    """
    Return a dict of {model_name: pipeline} for the three candidate models
    and a majority-class baseline.
    """
    models = {
        "Baseline (Majority)": build_pipeline(
            DummyClassifier(strategy="most_frequent", random_state=SEED),
            preprocessor,
        ),
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=SEED, C=1.0),
            preprocessor,
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=200, max_depth=6,
                min_samples_leaf=5, random_state=SEED, n_jobs=-1,
            ),
            preprocessor,
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1,
                max_depth=4, random_state=SEED,
            ),
            preprocessor,
        ),
    }
    return models


# ── Step 1: Run StratifiedKFold for all candidates ────────────────────────────

def run_kfold_comparison(models, X, y, n_splits=N_SPLITS):
    """
    Evaluate each model with StratifiedKFold and cross_val_score.
    Returns a dict: {name: {"fold_scores": [...], "mean": float, "std": float}}
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scorer = make_scorer(f1_score, zero_division=0)
    results = {}

    for name, pipeline in models.items():
        print(f"\n  Evaluating: {name}")
        try:
            scores = cross_val_score(
                pipeline, X, y,
                cv=cv, scoring=scorer, n_jobs=-1,
            )
            fold_info = {
                "fold_scores": scores.tolist(),
                "mean":        float(round(scores.mean(), 4)),
                "std":         float(round(scores.std(),  4)),
                "min":         float(round(scores.min(),  4)),
                "max":         float(round(scores.max(),  4)),
            }
            results[name] = fold_info
            print(f"    Fold F1 scores : {[round(s, 4) for s in scores]}")
            print(f"    Mean ± Std     : {fold_info['mean']:.4f} ± {fold_info['std']:.4f}")
        except Exception as exc:
            print(f"    [WARNING] cross_val_score failed for '{name}': {exc}")
            traceback.print_exc()
            results[name] = {
                "fold_scores": [],
                "mean": None, "std": None,
                "min": None,  "max": None,
                "error": str(exc),
            }

    return results


# ── Step 2: Nested CV for Gradient Boosting ───────────────────────────────────

def run_nested_cv(X, y, preprocessor, n_splits=N_SPLITS):
    """
    Nested cross-validation for Gradient Boosting:
      - Outer loop  : StratifiedKFold(5)  → honest generalisation estimate
      - Inner loop  : GridSearchCV(3-fold) → tunes max_depth
    Returns outer fold scores and the best inner params per fold.
    """
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=3,        shuffle=True, random_state=SEED)

    param_grid = {"clf__max_depth": [3, 4, 5, 6]}
    scorer = make_scorer(f1_score, zero_division=0)

    outer_scores  = []
    best_params   = []

    X_arr = X.values if hasattr(X, "values") else np.array(X)
    y_arr = y.values if hasattr(y, "values") else np.array(y)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_arr, y_arr), 1):
        try:
            X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
            y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

            # Rebuild preprocessor fresh for each fold to avoid leakage
            num_feats, cat_feats = get_feature_types(
                pd.DataFrame(X_tr, columns=X.columns)
            )
            prep_fold = get_preprocessor(num_feats, cat_feats)

            gb_pipe = Pipeline([
                ("prep", prep_fold),
                ("clf",  GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=SEED
                )),
            ])

            gs = GridSearchCV(
                gb_pipe, param_grid,
                cv=inner_cv, scoring=scorer, n_jobs=-1, refit=True,
            )
            X_tr_df = pd.DataFrame(X_tr, columns=X.columns)
            X_te_df = pd.DataFrame(X_te, columns=X.columns)
            gs.fit(X_tr_df, y_tr)

            preds = gs.predict(X_te_df)
            fold_f1 = f1_score(y_te, preds, zero_division=0)
            outer_scores.append(round(float(fold_f1), 4))
            best_params.append(gs.best_params_)
            print(f"    Fold {fold_idx}: F1={fold_f1:.4f}  best_params={gs.best_params_}")

        except Exception as exc:
            print(f"    [WARNING] Nested CV fold {fold_idx} failed: {exc}")
            traceback.print_exc()
            outer_scores.append(None)
            best_params.append({"error": str(exc)})

    valid_scores = [s for s in outer_scores if s is not None]
    mean_nested  = round(float(np.mean(valid_scores)), 4) if valid_scores else None
    std_nested   = round(float(np.std(valid_scores)),  4) if valid_scores else None

    return {
        "outer_fold_scores": outer_scores,
        "best_params_per_fold": best_params,
        "mean": mean_nested,
        "std":  std_nested,
    }


# ── Step 3: Choose the best generalising model ────────────────────────────────

def select_best_model(cv_results):
    """
    Pick the model with highest mean F1 and lowest std.
    Returns (best_name, reason_string).
    """
    valid = {
        name: info for name, info in cv_results.items()
        if info.get("mean") is not None
    }
    if not valid:
        return None, "No valid results to compare."

    # Primary: highest mean F1; tie-break: lowest std
    best_name = max(
        valid,
        key=lambda n: (valid[n]["mean"], -valid[n]["std"]),
    )
    info = valid[best_name]
    reason = (
        f"Highest mean F1 = {info['mean']:.4f} "
        f"with std = {info['std']:.4f} across {N_SPLITS} stratified folds."
    )
    return best_name, reason


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_fold_scores(cv_results, output_path):
    """Box-and-strip plot: per-fold F1 distribution for each model."""
    try:
        rows = []
        for name, info in cv_results.items():
            for s in info.get("fold_scores", []):
                rows.append({"Model": name, "F1": s})
        if not rows:
            print("  [WARNING] No fold scores to plot.")
            return

        df_plot = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, 5))
        palette = sns.color_palette("Set2", n_colors=len(cv_results))
        sns.boxplot(
            data=df_plot, x="Model", y="F1",
            palette=palette, width=0.45, linewidth=1.2, ax=ax,
        )
        sns.stripplot(
            data=df_plot, x="Model", y="F1",
            color="black", size=5, alpha=0.6, jitter=True, ax=ax,
        )
        ax.set_title(
            f"Task 16 — K-Fold F1 Distribution ({N_SPLITS}-Fold StratifiedKFold)",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved fold scores plot -> {output_path}")
    except Exception as exc:
        print(f"  [WARNING] fold scores plot failed: {exc}")
        traceback.print_exc()


def plot_mean_std_bar(cv_results, best_name, output_path):
    """Horizontal bar chart of mean F1 ± std for each model."""
    try:
        names   = list(cv_results.keys())
        means   = [cv_results[n].get("mean") or 0 for n in names]
        stds    = [cv_results[n].get("std")  or 0 for n in names]
        colors  = ["#e07b54" if n == best_name else "#5b8db8" for n in names]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(names, means, xerr=stds, color=colors,
                       edgecolor="white", height=0.55,
                       error_kw={"elinewidth": 2, "capsize": 4})
        ax.bar_label(bars, labels=[f"{m:.4f}" for m in means],
                     padding=6, fontsize=9)
        ax.set_xlabel("Mean F1-Score (± 1 std)", fontsize=11)
        ax.set_title(
            "Task 16 — Model Comparison: Mean F1 ± Std (K-Fold CV)",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor="#e07b54", label="Best model"),
            Patch(facecolor="#5b8db8", label="Other models"),
        ]
        ax.legend(handles=legend_elems, loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved mean±std bar chart -> {output_path}")
    except Exception as exc:
        print(f"  [WARNING] mean±std plot failed: {exc}")
        traceback.print_exc()


def plot_nested_cv(nested_info, output_path):
    """Bar chart for nested CV outer-fold scores."""
    try:
        scores = nested_info.get("outer_fold_scores", [])
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            print("  [WARNING] No nested CV scores to plot.")
            return

        folds  = [f"Fold {i+1}" for i, s in enumerate(scores) if s is not None]
        mean_v = nested_info.get("mean", 0) or 0

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(folds, valid_scores, color="#6c5ce7", edgecolor="white")
        ax.axhline(mean_v, color="crimson", linestyle="--", linewidth=1.8,
                   label=f"Mean = {mean_v:.4f}")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Outer Fold")
        ax.set_ylabel("F1 Score")
        ax.set_title(
            "Nested CV — Gradient Boosting Outer-Fold F1\n"
            "(Inner loop tunes max_depth via GridSearchCV)",
            fontsize=12,
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved nested CV plot -> {output_path}")
    except Exception as exc:
        print(f"  [WARNING] nested CV plot failed: {exc}")
        traceback.print_exc()


def plot_fold_heatmap(cv_results, output_path):
    """Heatmap: rows = models, cols = folds, values = F1."""
    try:
        data = {}
        max_folds = 0
        for name, info in cv_results.items():
            scores = info.get("fold_scores", [])
            if scores:
                data[name] = scores
                max_folds = max(max_folds, len(scores))

        if not data:
            print("  [WARNING] No data for heatmap.")
            return

        df_heat = pd.DataFrame(
            data,
            index=[f"Fold {i+1}" for i in range(max_folds)],
        ).T.round(4)

        fig, ax = plt.subplots(figsize=(max(8, max_folds * 1.5), len(data) * 0.8 + 1.5))
        sns.heatmap(
            df_heat, annot=True, fmt=".4f",
            cmap="YlGn", linewidths=0.5,
            vmin=0, vmax=1, ax=ax,
            cbar_kws={"label": "F1 Score"},
        )
        ax.set_title("Per-Fold F1 Scores — All Models", fontsize=13)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Model")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved fold heatmap -> {output_path}")
    except Exception as exc:
        print(f"  [WARNING] fold heatmap failed: {exc}")
        traceback.print_exc()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOGS_DIR,  exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Task 16 — Model Validation & K-Fold")
    print("  PlaceMux  Phase 1 Industry Immersion")
    print("=" * 65)

    # ── 0. Load data ─────────────────────────────────────────────────────────
    print("\n[0/6] Loading data ...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_data()
        # Combine train+val for cross-validation (test remains sealed)
        X_cv = pd.concat([X_train, X_val], ignore_index=True)
        y_cv = pd.concat([y_train, y_val], ignore_index=True)
        print(f"  CV pool  : {X_cv.shape}   "
              f"(class balance: {y_cv.mean():.2%} positive)")
        print(f"  Test set : {X_test.shape}  [sealed — not used in CV]")
    except Exception as exc:
        print(f"\n  [FATAL] Data loading failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── 1. Build preprocessor ─────────────────────────────────────────────────
    print("\n[1/6] Building shared preprocessor ...")
    try:
        numeric_features, categorical_features = get_feature_types(X_cv)
        preprocessor = get_preprocessor(numeric_features, categorical_features)
        print(f"  Numeric features     : {numeric_features}")
        print(f"  Categorical features : {categorical_features}")
    except Exception as exc:
        print(f"\n  [FATAL] Preprocessor setup failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── 2. K-Fold comparison across all models ───────────────────────────────
    print(f"\n[2/6] Running {N_SPLITS}-Fold StratifiedKFold on all candidate models ...")
    print(f"  Metric: {SCORING}  |  seed={SEED}  |  CV scheme: StratifiedKFold")
    try:
        models     = get_candidate_models(preprocessor)
        cv_results = run_kfold_comparison(models, X_cv, y_cv, n_splits=N_SPLITS)
    except Exception as exc:
        print(f"\n  [FATAL] K-Fold comparison failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── 3. Nested CV for Gradient Boosting ───────────────────────────────────
    print(f"\n[3/6] Running Nested CV for Gradient Boosting (outer={N_SPLITS}, inner=3) ...")
    print("  (Inner GridSearchCV tunes max_depth — outer folds give honest F1)")
    try:
        nested_info = run_nested_cv(X_cv, y_cv, preprocessor, n_splits=N_SPLITS)
        print(f"\n  Nested CV mean F1 : {nested_info['mean']:.4f}")
        print(f"  Nested CV std     : {nested_info['std']:.4f}")
        print("  (This is the fair estimate when GB params were also tuned)")
    except Exception as exc:
        print(f"\n  [WARNING] Nested CV failed: {exc}")
        traceback.print_exc()
        nested_info = {"outer_fold_scores": [], "mean": None, "std": None,
                       "best_params_per_fold": [], "error": str(exc)}

    # ── 4. Select best model ─────────────────────────────────────────────────
    print("\n[4/6] Selecting best generalising model ...")
    try:
        best_name, reason = select_best_model(cv_results)
        if best_name:
            print(f"\n  [BEST] BEST MODEL : {best_name}")
            print(f"         Reason     : {reason}")
        else:
            print("  [WARNING] Could not determine best model.")
    except Exception as exc:
        print(f"  [WARNING] Model selection failed: {exc}")
        best_name, reason = None, str(exc)

    # ── 5. Plots ─────────────────────────────────────────────────────────────
    print("\n[5/6] Generating plots ...")
    plot_fold_scores(
        cv_results,
        os.path.join(PLOTS_DIR, "fold_scores_boxplot.png"),
    )
    plot_mean_std_bar(
        cv_results, best_name,
        os.path.join(PLOTS_DIR, "mean_std_bar.png"),
    )
    plot_nested_cv(
        nested_info,
        os.path.join(PLOTS_DIR, "nested_cv_gb.png"),
    )
    plot_fold_heatmap(
        cv_results,
        os.path.join(PLOTS_DIR, "fold_heatmap.png"),
    )

    # ── 6. Save JSON report ──────────────────────────────────────────────────
    print("\n[6/6] Saving results JSON ...")
    try:
        report = {
            "task":       "Task 16 — Model Validation & K-Fold",
            "seed":       SEED,
            "n_splits":   N_SPLITS,
            "cv_scheme":  "StratifiedKFold",
            "metric":     SCORING,
            "models":     _safe_json(cv_results),
            "nested_cv_gradient_boosting": _safe_json(nested_info),
            "best_model": {
                "name":   best_name,
                "reason": reason,
            },
            "pitfalls_avoided": {
                "reporting_only_best_fold":
                    "All fold scores reported with mean ± std",
                "non_stratified_folds":
                    "StratifiedKFold used throughout to preserve class balance",
                "tuning_and_evaluating_same_folds":
                    "Nested CV separates tuning (inner) from evaluation (outer) for GB",
            },
        }
        with open(RESULTS_JSON, "w") as f:
            json.dump(report, f, indent=4)
        print(f"  Results saved -> {RESULTS_JSON}")
    except Exception as exc:
        print(f"  [ERROR] Failed to save JSON: {exc}")
        traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TASK 16 — COMPLETE")
    print("=" * 65)
    print(f"  CV Scheme  : StratifiedKFold  ({N_SPLITS} folds)")
    print(f"  Metric     : {SCORING.upper()}")
    print()
    print(f"  {'Model':<28} {'Mean F1':>9}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print(f"  {'-'*60}")
    for name, info in cv_results.items():
        m = info.get("mean")
        s = info.get("std")
        mn = info.get("min")
        mx = info.get("max")
        marker = " <-- BEST" if name == best_name else ""
        if m is not None:
            print(f"  {name:<28} {m:>9.4f}  {s:>7.4f}  {mn:>7.4f}  {mx:>7.4f}{marker}")
        else:
            print(f"  {name:<28}   {'ERROR':>9}{marker}")
    print()
    if nested_info.get("mean") is not None:
        print(f"  Nested CV GB (honest)  : "
              f"{nested_info['mean']:.4f} ± {nested_info['std']:.4f}")
    print(f"\n  Best generalising model: {best_name}")
    print(f"  Plots saved to        : {PLOTS_DIR}/")
    print(f"  Report saved to       : {RESULTS_JSON}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
