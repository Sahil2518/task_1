"""
Task 11 — Ensemble Learning
Defines diverse base models and ensemble builders (Voting + Stacking).
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    GradientBoostingClassifier,
)
from sklearn.pipeline import Pipeline
from src.config import CONFIG

# ──────────────────────────────────────────────
# 1.  Diverse Base Models
# ──────────────────────────────────────────────

def get_logistic_regression():
    """Linear model — captures linear decision boundary."""
    return LogisticRegression(
        max_iter=1000,
        random_state=CONFIG["seed"],
        C=1.0,
    )


def get_random_forest():
    """Bagging-based ensemble — captures non-linear splits."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=CONFIG["seed"],
        n_jobs=-1,
    )


def get_gradient_boosting():
    """Boosting model — focuses on hard examples sequentially."""
    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        random_state=CONFIG["seed"],
    )


# ──────────────────────────────────────────────
# 2.  Voting Ensemble (soft)
# ──────────────────────────────────────────────

def get_voting_ensemble(preprocessor):
    """
    Soft-voting over the 3 diverse classifiers.
    Each sub-pipeline applies the same preprocessor so that
    there is NO feature leakage between folds.
    """
    base_estimators = [
        ("lr",  Pipeline([("prep", preprocessor), ("clf", get_logistic_regression())])),
        ("rf",  Pipeline([("prep", preprocessor), ("clf", get_random_forest())])),
        ("gb",  Pipeline([("prep", preprocessor), ("clf", get_gradient_boosting())])),
    ]
    return VotingClassifier(estimators=base_estimators, voting="soft", n_jobs=-1)


# ──────────────────────────────────────────────
# 3.  Stacking Ensemble
# ──────────────────────────────────────────────

def get_stacking_ensemble(preprocessor):
    """
    Stacking with cross_val_predict (cv=5) to avoid fold leakage.
    Meta-learner: LogisticRegression on out-of-fold probabilities.
    Each base estimator is a full sub-pipeline (preprocessor + classifier).
    """
    base_estimators = [
        ("lr",  Pipeline([("prep", preprocessor), ("clf", get_logistic_regression())])),
        ("rf",  Pipeline([("prep", preprocessor), ("clf", get_random_forest())])),
        ("gb",  Pipeline([("prep", preprocessor), ("clf", get_gradient_boosting())])),
    ]
    meta_learner = LogisticRegression(max_iter=500, random_state=CONFIG["seed"])

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,                   # 5-fold OOF — prevents leakage
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )
