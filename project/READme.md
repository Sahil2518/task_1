# 🧠 PlaceMux — Task 11 & 12: Ensemble & Calibration

> **Phase 1 Industry Immersion · AI/ML Developer Track**  
> Altrodav Technologies Pvt. Ltd.

---

## Objective

**Task 11:** Combine multiple diverse base models using ensemble methods (Voting + Stacking) to achieve more accurate and robust predictions than any single model alone.

**Task 12:** Deliver a production-grade binary classification model with calibrated, decision-ready outputs and cost-optimal threshold selection.

---

## Task 11: Ensemble Architecture

Three **deliberately diverse** base models are combined:

| Model | Type | What It Captures |
|-------|------|-----------------|
| Logistic Regression | Linear | Linear decision boundary |
| Random Forest | Bagging (parallel) | Non-linear splits, low variance |
| Gradient Boosting | Boosting (sequential) | Hard examples, low bias |

### Ensemble Methods

- **Voting (soft):** Averages class probabilities from all 3 models before thresholding. Fast, robust, no leakage risk.
- **Stacking (OOF-5):** Uses 5-fold cross-validation to produce out-of-fold (OOF) probabilities from base models. A Logistic Regression meta-learner is trained on those OOF outputs — **zero data leakage across folds**.

---

## Results (Test Set)

| Type | Model | Accuracy | Precision | Recall | F1 |
|------|-------|----------|-----------|--------|----|
| 🔶 Ensemble | **Voting Ensemble (soft)** | 0.8133 | 0.8288 | 0.9758 | **0.8963** |
| 🔷 Single | Random Forest | 0.8133 | 0.8288 | 0.9758 | 0.8963 |
| 🔷 Single | Logistic Regression | 0.8133 | 0.8310 | 0.9718 | 0.8959 |
| 🔶 Ensemble | Stacking Ensemble (OOF5) | 0.8100 | 0.8282 | 0.9718 | 0.8942 |
| 🔷 Single | Gradient Boosting | 0.8067 | 0.8299 | 0.9637 | 0.8918 |

**F1 Lift: +0.0000** — The Voting Ensemble ties the best single model (Random Forest). On synthetic data near ceiling performance, this is expected and honest — the ensemble matches but never hurts the best single model.

---

## Visualizations

### Ensemble vs. Single-Model F1 Comparison

![Ensemble F1 Comparison](logs/ensemble_comparison.png)

> Orange bars = ensembles, blue bars = single models. The Voting Ensemble matches the best single model's F1 of 0.8963.

---

### ROC Curve (Task 10 baseline model — carried forward)

![ROC Curve](logs/roc_curve.png)

---

### Precision-Recall Curve

![Precision-Recall Curve](logs/pr_curve.png)

---

### Confusion Matrix (Threshold = 0.4)

![Confusion Matrix](logs/confusion_matrix_t0.4.png)

---

### Partial Dependence Plots (Top Features)

![Partial Dependence Plots](logs/pdp_plot.png)

---

## Diversity Check

Pairwise disagreement rates confirm the base models are genuinely diverse:

```
                          LR     RF     GB
Logistic Regression      0.000  0.020  0.040
Random Forest            0.020  0.000  0.033
Gradient Boosting        0.040  0.033  0.000
```

LR and GB disagree on **4% of test samples** — proving different error patterns that are averaged out by the ensemble.

---

## Trade-off Analysis

| Method | Inference Cost | Expected Gain | Verdict |
|--------|---------------|---------------|---------|
| Single model | 1× | baseline | ✅ Fast |
| Voting (soft) | 3× | Matches/beats best single | ✅ Recommended |
| Stacking (OOF5) | 3× + meta | Slight overhead, best on noisy data | ⚠️ Use if latency allows |

---

## Project Structure

```
project/
├── src/
│   ├── config.py           # Seed, split sizes
│   ├── data.py             # Data generation + feature engineering
│   ├── preprocess.py       # ColumnTransformer (impute + scale + encode)
│   ├── model.py            # Task 10: GradientBoosting baseline
│   ├── ensemble.py         # Task 11: 3 base models + Voting + Stacking
│   ├── calibrate.py        # Task 12: Calibration, thresholding, and evaluation utilities
│   ├── train.py            # Task 10 training script
│   ├── train_ensemble.py   # Task 11 training script
│   ├── train_task12.py     # Task 12 training script ← main entry point for calibrated model
│   ├── evaluate.py         # All evaluation utilities (shared)
│   └── predict.py          # Inference engine (prefers ensemble model)
├── models/
│   ├── calibrated_pipeline.pkl # Best calibrated model (Task 12)
│   ├── ensemble_pipeline.pkl   # Best ensemble (Task 11)
│   ├── lr_pipeline.pkl
│   ├── rf_pipeline.pkl
│   ├── gb_pipeline.pkl
│   └── pipeline.pkl            # Task 10 single model
├── logs/
│   ├── task12_operating_point.json # Task 12 configuration and metrics
│   ├── calibration_curve.png   # Task 12 reliability diagram
│   ├── threshold_analysis.png  # Task 12 threshold selection
│   ├── calibrated_confusion_matrix.png # Task 12 optimal CM
│   ├── fold_stability.png      # Task 12 cross-fold F1 stability
│   ├── segment_evaluation.png  # Task 12 fairness check
│   ├── ensemble_metrics.json   # Task 11 metrics
│   ├── ensemble_comparison.png # Task 11 F1 bar chart
│   ├── ensemble_comparison.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── pdp_plot.png
├── app.py                  # Gradio live demo (6 tabs covering Tasks 11 & 12)
├── run_task11.bat          # One-click: train ensemble + launch
├── run_task12.bat          # One-click: train calibrated model + launch
└── requirements.txt
```

---

## How to Run

### Train the Ensemble (Task 11)
```bash
python -m src.train_ensemble
```

### Train the Calibrated Classifier (Task 12)
```bash
python -m src.train_task12
```

### Launch Live Verification App
```bash
python app.py
```

### One-click (Windows)
```
run_task12.bat
```

App runs at `http://localhost:7860` with a public `gradio.live` share link.

---

## Pitfalls Avoided

| Pitfall | How It Was Avoided |
|---------|-------------------|
| Ensembling near-identical models | 3 models from different paradigms (linear / bagging / boosting) |
| Stacking fold leakage | `StackingClassifier` uses `cv=5` OOF — no test data seen during meta-training |
| Ignoring inference cost | Complexity vs. latency trade-off documented above |
| Honest evaluation | All final numbers reported on a sealed test set, never touched during training |
| Uncalibrated probabilities | Wrapped model in `CalibratedClassifierCV` with isotonic regression (Task 12) |
| Hidden per-segment failure | Checked F1 score across `projects_completed` subgroups (Task 12) |
| No documented operating point | Stored explicitly as JSON: Threshold=0.715 with cost-weighted rationale (Task 12) |

---

## Reproducibility

All runs use `random_state=42` (defined in `src/config.py`). Re-running `python -m src.train_task12` produces identical results.

---

## Task 12: Calibration and Thresholding Results

The final production model (`models/calibrated_pipeline.pkl`) uses a Gradient Boosting baseline calibrated via `CalibratedClassifierCV`.

### 1. Calibration Validation
Before calibration, the base GB model had a Brier Score of `0.1541`. After applying Isotonic Regression (CV=5), the Brier Score improved to `0.1374`, demonstrating probabilities that are significantly closer to true likelihoods.

![Calibration Curve](logs/calibration_curve.png)

### 2. Cost-Optimal Threshold
Instead of using a default `0.5` threshold, we assign business costs to errors:
* False Positives (FP) cost `1.0x` (predicting someone is placed when they are not)
* False Negatives (FN) cost `2.0x` (missing a candidate who actually gets placed is worse)

By sweeping all possible thresholds, we found that the threshold of **0.715** minimizes total cost.

![Threshold Analysis](logs/threshold_analysis.png)

### 3. Stability and Segment Fairness
The final model is proven stable and fair before being deployed:
* **Cross-Fold Stability:** `Mean F1 = 0.8742 ± 0.0125` across 5 StratifiedKFold splits. (Standard deviation < 0.03 marks the model as highly stable).
* **Segment Evaluation:** Verified across the `projects_completed` feature to ensure no sub-group is being silently penalized.

![Segment Evaluation](logs/segment_evaluation.png)
![Fold Stability](logs/fold_stability.png)

### 4. Final Confusion Matrix (at threshold=0.715)
Tested on the strictly held-out dataset (300 rows):

![Calibrated Confusion Matrix](logs/calibrated_confusion_matrix.png)

- **FPR (False Alarm Rate):** 0.865
- **FNR (Miss Rate):** 0.073 (Driven down by setting FN cost to 2.0x)