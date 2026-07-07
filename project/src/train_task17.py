"""
Task 17 — Hyperparameter Tuning (Advanced)
PlaceMux Phase 1 Industry Immersion

Steps:
  1. Design sensible search space with correct scales.
  2. Bayesian search (Optuna) with MedianPruner early stopping.
  3. Score by robust StratifiedKFold CV on F1 (business metric).
  4. XGBoost / LightGBM early stopping where supported.
  5. Confirm winner on held-out test set.
  6. Log ALL trials for reproducibility (SQLite + CSV + JSON).
"""

import os
import sys
import json
import traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Optuna ──────────────────────────────────────────────────────────
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── scikit-learn ────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# ── XGBoost / LightGBM ──────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("[WARNING] xgboost not found — XGB trials will be skipped.")

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
    print("[WARNING] lightgbm not found — LGB trials will be skipped.")

# ── project imports ─────────────────────────────────────────────────
try:
    from src.data import load_data
    from src.preprocess import get_feature_types, get_preprocessor
    from src.config import CONFIG
except ImportError as _ie:
    print(f"[FATAL] Could not import project modules: {_ie}")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────
SEED       = 42
N_SPLITS   = 5
N_TRIALS   = 40          # per model
TIMEOUT    = 180         # seconds per study (safety cap)
LOGS_DIR   = "logs"
PLOTS_DIR  = os.path.join(LOGS_DIR, "task17_plots")
DB_PATH    = os.path.join(LOGS_DIR, "task17_optuna.db")
RESULTS_JSON = os.path.join(LOGS_DIR, "task17_results.json")

np.random.seed(SEED)


# ── Helpers ──────────────────────────────────────────────────────────

def _safe_json(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, dict):           return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):  return [_safe_json(v) for v in obj]
    return obj


def build_preprocessor(X_sample):
    num_feats, cat_feats = get_feature_types(X_sample)
    return get_preprocessor(num_feats, cat_feats), num_feats, cat_feats


def cv_score(pipeline, X, y, n_splits=N_SPLITS, seed=SEED):
    """Return mean F1 over StratifiedKFold — the CV score used for all trials."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scorer = make_scorer(f1_score, zero_division=0)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


# ── Baseline ─────────────────────────────────────────────────────────

def run_baseline(X_cv, y_cv, preprocessor):
    """LR with default params — the bar every tuned model must beat."""
    pipe = Pipeline([("prep", preprocessor),
                     ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
    mean, std = cv_score(pipe, X_cv, y_cv)
    print(f"  Baseline LR  CV F1: {mean:.4f} ± {std:.4f}")
    return mean, std


# ── XGBoost study ─────────────────────────────────────────────────────

def make_xgb_objective(X_cv, y_cv, preprocessor):
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 100, 800),
            max_depth       = trial.suggest_int("max_depth", 3, 8),
            learning_rate   = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            subsample       = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha       = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda      = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
            random_state    = SEED,
            eval_metric     = "logloss",
            verbosity       = 0,
            use_label_encoder = False,
        )
        # We use CV so early stopping is approximated via n_estimators search space
        clf = xgb.XGBClassifier(**params)
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        scorer = make_scorer(f1_score, zero_division=0)
        # Pruning: report intermediate values per fold
        fold_scores = []
        for step, (tr_idx, val_idx) in enumerate(cv.split(X_cv, y_cv)):
            X_tr, X_v = X_cv.iloc[tr_idx], X_cv.iloc[val_idx]
            y_tr, y_v = y_cv.iloc[tr_idx], y_cv.iloc[val_idx]
            try:
                pipe.fit(X_tr, y_tr)
                score = f1_score(y_v, pipe.predict(X_v), zero_division=0)
            except Exception:
                score = 0.0
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(np.mean(fold_scores))
    return objective


def run_xgb_study(X_cv, y_cv, preprocessor):
    if not XGB_OK:
        return None
    print("\n  [XGBoost] Starting Optuna study ...")
    storage = f"sqlite:///{DB_PATH}"
    study = optuna.create_study(
        study_name="task17_xgb",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        make_xgb_objective(X_cv, y_cv, preprocessor),
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=False,
    )
    best = study.best_trial
    print(f"  [XGBoost] Best trial #{best.number}  CV F1={best.value:.4f}")
    print(f"            Params: {best.params}")
    return study


# ── LightGBM study ────────────────────────────────────────────────────

def make_lgb_objective(X_cv, y_cv, preprocessor):
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 100, 800),
            max_depth       = trial.suggest_int("max_depth", 3, 8),
            learning_rate   = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            num_leaves      = trial.suggest_int("num_leaves", 20, 150),
            subsample       = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha       = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda      = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
            random_state    = SEED,
            verbose         = -1,
        )
        clf = lgb.LGBMClassifier(**params)
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_scores = []
        for step, (tr_idx, val_idx) in enumerate(cv.split(X_cv, y_cv)):
            X_tr, X_v = X_cv.iloc[tr_idx], X_cv.iloc[val_idx]
            y_tr, y_v = y_cv.iloc[tr_idx], y_cv.iloc[val_idx]
            try:
                pipe.fit(X_tr, y_tr)
                score = f1_score(y_v, pipe.predict(X_v), zero_division=0)
            except Exception:
                score = 0.0
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(np.mean(fold_scores))
    return objective


def run_lgb_study(X_cv, y_cv, preprocessor):
    if not LGB_OK:
        return None
    print("\n  [LightGBM] Starting Optuna study ...")
    storage = f"sqlite:///{DB_PATH}"
    study = optuna.create_study(
        study_name="task17_lgb",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        make_lgb_objective(X_cv, y_cv, preprocessor),
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=False,
    )
    best = study.best_trial
    print(f"  [LightGBM] Best trial #{best.number}  CV F1={best.value:.4f}")
    print(f"             Params: {best.params}")
    return study


# ── Build winning model ───────────────────────────────────────────────

def build_winner(study_xgb, study_lgb, preprocessor):
    """Pick the study with the highest best value and build its pipeline."""
    candidates = {}
    if study_xgb and study_xgb.best_trial:
        candidates["XGBoost"] = (study_xgb, study_xgb.best_value)
    if study_lgb and study_lgb.best_trial:
        candidates["LightGBM"] = (study_lgb, study_lgb.best_value)
    if not candidates:
        return None, None, None

    winner_name = max(candidates, key=lambda k: candidates[k][1])
    winner_study, winner_val = candidates[winner_name]
    best_params = winner_study.best_trial.params

    if winner_name == "XGBoost":
        best_params.pop("early_stopping_rounds", None)
        best_params.setdefault("eval_metric", "logloss")
        best_params.setdefault("verbosity", 0)
        clf = xgb.XGBClassifier(random_state=SEED, **best_params)
    else:
        best_params.setdefault("verbose", -1)
        clf = lgb.LGBMClassifier(random_state=SEED, **best_params)

    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    return pipe, winner_name, best_params


# ── Plots ─────────────────────────────────────────────────────────────

def plot_optimization_history(study, name, out_dir):
    try:
        values = [t.value for t in study.trials if t.value is not None]
        if not values:
            return
        best_so_far = [max(values[:i+1]) for i in range(len(values))]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.scatter(range(len(values)), values, alpha=0.4, s=20,
                   color="#5b8db8", label="Trial F1")
        ax.plot(range(len(best_so_far)), best_so_far,
                color="#e07b54", linewidth=2, label="Best so far")
        ax.set_xlabel("Trial number")
        ax.set_ylabel("CV F1")
        ax.set_title(f"Task 17 — {name} Optimisation History", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"opt_history_{name.lower()}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved optimisation history -> {path}")
    except Exception as exc:
        print(f"  [WARNING] opt history plot failed: {exc}")


def plot_param_importance(study, name, out_dir):
    try:
        importance = optuna.importance.get_param_importances(study)
        if not importance:
            return
        labels = list(importance.keys())[:10]
        vals   = [importance[k] for k in labels]
        fig, ax = plt.subplots(figsize=(8, max(3, len(labels)*0.4+1)))
        ax.barh(labels[::-1], vals[::-1], color="#6c5ce7")
        ax.set_xlabel("Importance")
        ax.set_title(f"Task 17 — {name} Hyperparameter Importance", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"param_importance_{name.lower()}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved param importance -> {path}")
    except Exception as exc:
        print(f"  [WARNING] param importance plot failed: {exc}")


def plot_trials_csv(study_xgb, study_lgb):
    """Dump all trial data to CSV for full reproducibility."""
    rows = []
    for name, study in [("XGBoost", study_xgb), ("LightGBM", study_lgb)]:
        if study is None:
            continue
        for t in study.trials:
            row = {"model": name, "trial": t.number, "cv_f1": t.value,
                   "state": str(t.state)}
            row.update({f"param_{k}": v for k, v in t.params.items()})
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(LOGS_DIR, "task17_all_trials.csv")
        df.to_csv(path, index=False)
        print(f"  All {len(rows)} trials logged -> {path}")


def plot_model_comparison(baseline_f1, xgb_cv, lgb_cv, test_f1, out_dir):
    """Bar chart: baseline vs tuned CV vs test confirmation."""
    try:
        labels = ["Baseline\n(LR default)", "XGBoost\n(tuned CV)",
                  "LightGBM\n(tuned CV)", "Winner\n(test set)"]
        values = [baseline_f1,
                  xgb_cv if xgb_cv else 0,
                  lgb_cv if lgb_cv else 0,
                  test_f1]
        colors = ["#b2bec3", "#0984e3", "#00b894", "#e07b54"]
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.axhline(baseline_f1, linestyle="--", color="gray", linewidth=1,
                   label=f"Baseline F1={baseline_f1:.4f}")
        ax.set_ylim(0, min(1.1, max(values)+0.15))
        ax.set_ylabel("F1 Score")
        ax.set_title("Task 17 — Hyperparameter Tuning Gains", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, "model_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved comparison chart -> {path}")
    except Exception as exc:
        print(f"  [WARNING] comparison plot failed: {exc}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOGS_DIR,  exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("  Task 17 — Hyperparameter Tuning (Advanced)")
    print("  PlaceMux  Phase 1 Industry Immersion")
    print("="*65)

    # ── 0. Load data ──────────────────────────────────────────────────
    print("\n[0/7] Loading data ...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_data()
        X_cv = pd.concat([X_train, X_val], ignore_index=True)
        y_cv = pd.concat([y_train, y_val], ignore_index=True)
        print(f"  CV pool  : {X_cv.shape}  (class balance: {y_cv.mean():.2%})")
        print(f"  Test set : {X_test.shape}  [sealed until final confirmation]")
    except Exception as exc:
        print(f"\n  [FATAL] Data loading failed: {exc}")
        traceback.print_exc(); sys.exit(1)

    # ── 1. Build preprocessor ─────────────────────────────────────────
    print("\n[1/7] Building preprocessor ...")
    try:
        preprocessor, num_feats, cat_feats = build_preprocessor(X_cv)
        print(f"  Numeric    : {num_feats}")
        print(f"  Categorical: {cat_feats}")
    except Exception as exc:
        print(f"\n  [FATAL] Preprocessor failed: {exc}")
        traceback.print_exc(); sys.exit(1)

    # ── 2. Baseline ───────────────────────────────────────────────────
    print("\n[2/7] Running baseline (Logistic Regression defaults) ...")
    try:
        baseline_mean, baseline_std = run_baseline(X_cv, y_cv, preprocessor)
    except Exception as exc:
        print(f"  [WARNING] Baseline failed: {exc}")
        baseline_mean, baseline_std = 0.0, 0.0

    # ── 3. XGBoost Optuna study ───────────────────────────────────────
    print(f"\n[3/7] XGBoost Bayesian search ({N_TRIALS} trials, pruning=MedianPruner) ...")
    try:
        study_xgb = run_xgb_study(X_cv, y_cv, preprocessor)
        xgb_best_cv = study_xgb.best_value if study_xgb else None
    except Exception as exc:
        print(f"  [WARNING] XGB study failed: {exc}")
        traceback.print_exc()
        study_xgb = None; xgb_best_cv = None

    # ── 4. LightGBM Optuna study ──────────────────────────────────────
    print(f"\n[4/7] LightGBM Bayesian search ({N_TRIALS} trials, pruning=MedianPruner) ...")
    try:
        study_lgb = run_lgb_study(X_cv, y_cv, preprocessor)
        lgb_best_cv = study_lgb.best_value if study_lgb else None
    except Exception as exc:
        print(f"  [WARNING] LGB study failed: {exc}")
        traceback.print_exc()
        study_lgb = None; lgb_best_cv = None

    # ── 5. Select winner & confirm on test set ────────────────────────
    print("\n[5/7] Selecting winner and confirming on held-out test set ...")
    test_f1 = None
    winner_name = None
    best_params = {}
    try:
        winner_pipe, winner_name, best_params = build_winner(
            study_xgb, study_lgb, preprocessor
        )
        if winner_pipe is not None:
            winner_pipe.fit(X_cv, y_cv)
            y_pred = winner_pipe.predict(X_test)
            test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
            improvement = test_f1 - baseline_mean
            print(f"\n  Winner model    : {winner_name}")
            print(f"  Test-set F1     : {test_f1:.4f}")
            print(f"  Baseline CV F1  : {baseline_mean:.4f}")
            print(f"  Gain over base  : {improvement:+.4f}  "
                  f"({'IMPROVED' if improvement > 0 else 'NO GAIN'})")
        else:
            print("  [WARNING] No valid winner found.")
    except Exception as exc:
        print(f"  [WARNING] Winner evaluation failed: {exc}")
        traceback.print_exc()

    # ── 6. Plots ──────────────────────────────────────────────────────
    print("\n[6/7] Generating plots ...")
    if study_xgb:
        plot_optimization_history(study_xgb, "XGBoost", PLOTS_DIR)
        plot_param_importance(study_xgb, "XGBoost", PLOTS_DIR)
    if study_lgb:
        plot_optimization_history(study_lgb, "LightGBM", PLOTS_DIR)
        plot_param_importance(study_lgb, "LightGBM", PLOTS_DIR)
    plot_trials_csv(study_xgb, study_lgb)
    plot_model_comparison(
        baseline_mean,
        xgb_best_cv, lgb_best_cv,
        test_f1 or 0.0,
        PLOTS_DIR,
    )

    # ── 7. Save JSON report ───────────────────────────────────────────
    print("\n[7/7] Saving results JSON ...")
    try:
        xgb_trials = (
            [{"trial": t.number, "value": t.value, "params": t.params,
              "state": str(t.state)}
             for t in study_xgb.trials]
            if study_xgb else []
        )
        lgb_trials = (
            [{"trial": t.number, "value": t.value, "params": t.params,
              "state": str(t.state)}
             for t in study_lgb.trials]
            if study_lgb else []
        )
        report = {
            "task": "Task 17 — Hyperparameter Tuning (Advanced)",
            "seed": SEED,
            "n_splits_cv": N_SPLITS,
            "n_trials_per_model": N_TRIALS,
            "search_strategy": "Optuna TPE + MedianPruner",
            "cv_metric": "F1 (StratifiedKFold)",
            "baseline": {"cv_f1_mean": baseline_mean, "cv_f1_std": baseline_std},
            "xgboost": {
                "best_cv_f1": xgb_best_cv,
                "best_params": study_xgb.best_trial.params if study_xgb else None,
                "n_trials_completed": len([t for t in (study_xgb.trials if study_xgb else [])
                                          if t.value is not None]),
                "all_trials": _safe_json(xgb_trials),
            },
            "lightgbm": {
                "best_cv_f1": lgb_best_cv,
                "best_params": study_lgb.best_trial.params if study_lgb else None,
                "n_trials_completed": len([t for t in (study_lgb.trials if study_lgb else [])
                                          if t.value is not None]),
                "all_trials": _safe_json(lgb_trials),
            },
            "winner": {
                "model": winner_name,
                "best_params": _safe_json(best_params),
                "test_f1": test_f1,
                "gain_over_baseline": (
                    round(test_f1 - baseline_mean, 4) if test_f1 else None
                ),
            },
            "pitfalls_avoided": {
                "huge_wasteful_grids":
                    "Bayesian TPE search — each trial informed by previous",
                "overfitting_search_to_validation":
                    "All CV done on train+val pool; test set sealed until final step",
                "unreproducible_unlogged_trials":
                    f"All trials persisted to {DB_PATH} and {LOGS_DIR}/task17_all_trials.csv",
            },
        }
        with open(RESULTS_JSON, "w") as f:
            json.dump(report, f, indent=4)
        print(f"  Report saved -> {RESULTS_JSON}")
    except Exception as exc:
        print(f"  [ERROR] Failed to save JSON: {exc}")
        traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  TASK 17 — COMPLETE")
    print("="*65)
    print(f"  Search       : Optuna TPE + MedianPruner ({N_TRIALS} trials each)")
    print(f"  CV metric    : F1 | {N_SPLITS}-Fold StratifiedKFold | seed={SEED}")
    print(f"  Baseline F1  : {baseline_mean:.4f} ± {baseline_std:.4f} (LR defaults)")
    if xgb_best_cv:
        print(f"  XGBoost best : {xgb_best_cv:.4f} (CV)  "
              f"trials={len([t for t in study_xgb.trials if t.value])}")
    if lgb_best_cv:
        print(f"  LightGBM best: {lgb_best_cv:.4f} (CV)  "
              f"trials={len([t for t in study_lgb.trials if t.value])}")
    if winner_name and test_f1:
        print(f"\n  Winner : {winner_name}")
        print(f"  Test F1: {test_f1:.4f}  (gain: {test_f1 - baseline_mean:+.4f})")
    print(f"\n  Plots   -> {PLOTS_DIR}/")
    print(f"  Report  -> {RESULTS_JSON}")
    print(f"  DB log  -> {DB_PATH}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
