import gradio as gr
import pandas as pd
import numpy as np
import json
import os
import sys
import joblib

from src.predict import load_artifacts
from src.data import feature_engineering


# ──────────────────────────────────────────────────────────────────────────────
# Core prediction helper
# ──────────────────────────────────────────────────────────────────────────────

def predict_candidate(reg_date, login_date, domain, aptitude, projects):
    try:
        pipeline = load_artifacts(prefer_ensemble=True)

        df_raw = pd.DataFrame([{
            "registration_date": reg_date,
            "last_login_date":   login_date,
            "domain_score":      domain,
            "aptitude_score":    aptitude,
            "projects_completed": projects,
            "has_offer_letter":  np.nan,
            "days_to_placement": np.nan,
            "random_noise":      np.nan,
        }])

        df_raw["registration_date"] = pd.to_datetime(df_raw["registration_date"])
        df_raw["last_login_date"]   = pd.to_datetime(df_raw["last_login_date"])

        df_engineered = feature_engineering(df_raw, prune_leaks=True)
        if "placed" in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=["placed"])

        pred = pipeline.predict(df_engineered)[0]
        prob = (
            pipeline.predict_proba(df_engineered)[0][1]
            if hasattr(pipeline, "predict_proba")
            else None
        )

        result = "✅ Placed" if pred == 1 else "❌ Not Placed"
        if prob is not None:
            result += f"  (Probability: {prob:.2%})"
        return result

    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ──────────────────────────────────────────────────────────────────────────────
# Load ensemble metrics summary for the dashboard tab
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics_summary():
    path = "logs/ensemble_metrics.json"
    if not os.path.exists(path):
        return "⚠️ Run `python -m src.train_ensemble` to generate metrics.", "", ""

    try:
        with open(path) as f:
            m = json.load(f)
    except json.JSONDecodeError:
        return "⚠️ Error: `ensemble_metrics.json` is corrupted.", "", ""

    best_single = m.get("best_single_model", "N/A")
    best_ens    = m.get("best_ensemble",     "N/A")
    lift        = m.get("lift_f1", 0.0)

    # Build a markdown table of all models
    rows = []
    for name, scores in {**m.get("single_models", {}), **m.get("ensembles", {})}.items():
        tag = "🔶 Ensemble" if name in m.get("ensembles", {}) else "🔷 Single"
        rows.append(
            f"| {tag} | **{name}** | {scores['accuracy']:.4f} | "
            f"{scores['precision']:.4f} | {scores['recall']:.4f} | {scores['f1']:.4f} |"
        )

    table = (
        "| Type | Model | Accuracy | Precision | Recall | F1 |\n"
        "|------|-------|----------|-----------|--------|----|\n"
        + "\n".join(rows)
    )

    summary = (
        f"**Best single model:** {best_single}\n\n"
        f"**Best ensemble:** {best_ens}\n\n"
        f"**F1 Lift (ensemble vs best single):** {lift:+.4f}"
    )

    return summary, table, f"Lift: {lift:+.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Task 12 helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_task12_metrics():
    path = "logs/task12_operating_point.json"
    if not os.path.exists(path):
        return "⚠️ Run `python -m src.train_task12` to generate Task 12 metrics.", ""

    try:
        with open(path) as f:
            m = json.load(f)
    except json.JSONDecodeError:
        return "⚠️ Error: `task12_operating_point.json` is corrupted.", ""

    tm = m.get("test_metrics", {})
    stab = m.get("cross_fold_stability", {})
    err = m.get("expected_error_rates", {})

    summary = (
        f"**Operating threshold:** `{m.get('operating_threshold', 'N/A'):.3f}`  "
        f"(cost weights: FP×{m['cost_weights']['FP']} | FN×{m['cost_weights']['FN']})\n\n"
        f"**Calibration:** {m.get('calibration_method', '')}  "
        f"| Brier before: `{m.get('brier_score_uncalibrated', 0):.4f}`  "
        f"→ after: `{m.get('brier_score_calibrated', 0):.4f}`\n\n"
        f"**Test F1:** `{tm.get('f1', 0):.4f}`  "
        f"| ROC-AUC: `{tm.get('roc_auc', 0):.4f}`  "
        f"| Brier: `{tm.get('brier_score', 0):.4f}`\n\n"
        f"**Cross-fold stability:** Mean F1 = `{stab.get('mean_f1', 0):.4f}` ± `{stab.get('std_f1', 0):.4f}`  "
        f"({'✅ Stable' if stab.get('stable') else '⚠️ Unstable'})\n\n"
        f"**Error rates:** FPR = `{err.get('FPR (false alarm rate)', 0):.3f}`  "
        f"| FNR = `{err.get('FNR (miss rate)', 0):.3f}`"
    )

    # Build segment table
    seg_rows = m.get("segment_evaluation", [])
    if seg_rows:
        table = (
            "| Segment | N | Precision | Recall | F1 |\n"
            "|---------|---|-----------|--------|----|\n"
        )
        for r in seg_rows:
            table += f"| {r['Segment']} | {r['N']} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} |\n"
    else:
        table = "_No segment data available._"

    return summary, table


def predict_candidate_calibrated(reg_date, login_date, domain, aptitude, projects, threshold):
    """Live prediction using the Task 12 calibrated model at user-chosen threshold."""
    cal_path = "models/calibrated_pipeline.pkl"
    if not os.path.exists(cal_path):
        return "⚠️ Run `python -m src.train_task12` first to build the calibrated model."
    try:
        from src.data import feature_engineering
        pipeline = joblib.load(cal_path)

        df_raw = pd.DataFrame([{
            "registration_date": reg_date,
            "last_login_date":   login_date,
            "domain_score":      domain,
            "aptitude_score":    aptitude,
            "projects_completed": projects,
            "has_offer_letter":  np.nan,
            "days_to_placement": np.nan,
            "random_noise":      np.nan,
        }])
        df_raw["registration_date"] = pd.to_datetime(df_raw["registration_date"])
        df_raw["last_login_date"]   = pd.to_datetime(df_raw["last_login_date"])

        df_eng = feature_engineering(df_raw, prune_leaks=True)
        if "placed" in df_eng.columns:
            df_eng = df_eng.drop(columns=["placed"])

        prob = pipeline.predict_proba(df_eng)[0][1]
        pred = int(prob >= threshold)

        result = "✅ Placed" if pred == 1 else "❌ Not Placed"
        result += f"  |  Calibrated Probability: {prob:.2%}  |  Threshold used: {threshold:.2f}"
        return result
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


with gr.Blocks(title="PlaceMux — Task 11 & 12") as demo:

    gr.Markdown(
        """
        # 🧠 PlaceMux Placement Predictor — Task 11 · 12
        **Live verification UI.** Task 11: Ensemble Learning · Task 12: Calibrated Binary Classification
        """,
        elem_classes=["header-box"],
    )

    with gr.Tabs():

        # ── Tab 1: Live Prediction ─────────────────────────────────
        with gr.Tab("🎯 Predict Placement"):
            gr.Markdown("Enter a candidate's details below to get the ensemble prediction.")
            with gr.Row():
                reg_date   = gr.Textbox(label="Registration Date (YYYY-MM-DD)", value="2026-01-15")
                login_date = gr.Textbox(label="Last Login Date (YYYY-MM-DD)",   value="2026-03-20")
            with gr.Row():
                domain   = gr.Number(label="Domain Score (0-100)",    value=85.5)
                aptitude = gr.Number(label="Aptitude Score (0-100)",   value=90.0)
                projects = gr.Number(label="Projects Completed",       value=4)

            predict_btn = gr.Button("🚀 Run Ensemble Prediction", variant="primary", size="lg")
            output = gr.Textbox(
                label="Ensemble Prediction Result",
                text_align="center",
                elem_classes=["result-box"],
            )

            predict_btn.click(
                predict_candidate,
                inputs=[reg_date, login_date, domain, aptitude, projects],
                outputs=output,
            )

            gr.Markdown(
                """
                ### 📌 Example Candidates
                | Scenario | Domain | Aptitude | Projects | Expected |
                |----------|--------|----------|----------|----------|
                | Strong candidate | 90 | 92 | 6 | ✅ Placed |
                | Borderline | 65 | 70 | 2 | ⚖️ Uncertain |
                | Weak candidate | 40 | 45 | 0 | ❌ Not Placed |
                """
            )

        # ── Tab 2: Ensemble Dashboard ──────────────────────────────
        with gr.Tab("📊 Ensemble Dashboard"):
            gr.Markdown("### Task 11 — Ensemble vs. Single-Model Results (Test Set)")

            refresh_btn = gr.Button("🔄 Load Latest Metrics", variant="secondary")
            lift_display = gr.Textbox(label="Ensemble Lift Summary", lines=4)
            model_table  = gr.Markdown()
            lift_badge   = gr.Textbox(label="F1 Lift at a Glance")

            def refresh_dashboard():
                summary, table, badge = load_metrics_summary()
                return summary, table, badge

            refresh_btn.click(refresh_dashboard, outputs=[lift_display, model_table, lift_badge])

            # Auto-load on app start
            demo.load(refresh_dashboard, outputs=[lift_display, model_table, lift_badge])

            if os.path.exists("logs/ensemble_comparison.png"):
                gr.Image("logs/ensemble_comparison.png", label="F1 Comparison Chart")

        # ── Tab 3: How It Works ────────────────────────────────────
        with gr.Tab("📖 How It Works"):
            gr.Markdown(
                """
                ## Task 11 — Ensemble Architecture

                ### Base Models (Diverse by Design)
                | Model | Type | Captures |
                |-------|------|---------|
                | **Logistic Regression** | Linear | Linear decision boundary |
                | **Random Forest** | Bagging | Non-linear splits, low variance |
                | **Gradient Boosting** | Boosting | Hard examples, low bias |

                ### Ensemble Methods
                **Voting (soft):** Each model outputs a probability; the ensemble
                averages them before thresholding. Simple, fast, robust.

                **Stacking (OOF-5):** Base models are trained with 5-fold
                cross-validation to produce out-of-fold (OOF) probabilities.
                A LogisticRegression meta-learner is trained on those OOF
                probabilities — **no data leakage across folds**.

                ### Why Diversity Matters
                Ensembling near-identical models gains nothing.
                The **pairwise disagreement matrix** (printed in the training log)
                confirms that each base model makes different errors, so averaging
                them out actually helps.

                ### Complexity vs. Latency Trade-off
                | Method | Inference cost | Expected lift |
                |--------|---------------|---------------|
                | Single model | 1× | baseline |
                | Voting | 3× | +moderate |
                | Stacking | 3× + meta | +moderate/high |

                Stacking is slightly slower at inference but typically offers
                the highest lift; use Voting if latency is critical.
                """
            )

        # ── Tab 4: Task 12 — Calibrated Classifier ─────────────────
        with gr.Tab("🎯 Task 12 — Calibrated Prediction"):
            gr.Markdown("### Task 12 — Live Prediction with Calibrated Probabilities")
            gr.Markdown(
                "Uses the **CalibratedClassifierCV** model.  "
                "Adjust the threshold to simulate different operating points."
            )
            with gr.Row():
                t12_reg_date   = gr.Textbox(label="Registration Date (YYYY-MM-DD)", value="2026-01-15")
                t12_login_date = gr.Textbox(label="Last Login Date (YYYY-MM-DD)",   value="2026-03-20")
            with gr.Row():
                t12_domain   = gr.Number(label="Domain Score (0-100)",   value=85.5)
                t12_aptitude = gr.Number(label="Aptitude Score (0-100)",  value=90.0)
                t12_projects = gr.Number(label="Projects Completed",      value=4)
            t12_threshold = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, value=0.40,
                label="Decision Threshold (default = cost-optimal)"
            )
            t12_btn = gr.Button("🎯 Run Calibrated Prediction", variant="primary", size="lg")
            t12_output = gr.Textbox(label="Calibrated Prediction Result", text_align="center")
            t12_btn.click(
                predict_candidate_calibrated,
                inputs=[t12_reg_date, t12_login_date, t12_domain, t12_aptitude, t12_projects, t12_threshold],
                outputs=t12_output,
            )

        # ── Tab 5: Task 12 — Dashboard ─────────────────────────────
        with gr.Tab("📊 Task 12 — Calibration Dashboard"):
            gr.Markdown("### Task 12 — Calibration & Threshold Results")

            t12_refresh = gr.Button("🔄 Load Task 12 Metrics", variant="secondary")
            t12_summary = gr.Markdown()
            t12_seg_tbl = gr.Markdown(label="Segment F1 Table")

            def refresh_task12():
                summary, table = load_task12_metrics()
                return summary, table

            t12_refresh.click(refresh_task12, outputs=[t12_summary, t12_seg_tbl])
            demo.load(refresh_task12, outputs=[t12_summary, t12_seg_tbl])

            for img_path, img_label in [
                ("logs/calibration_curve.png",        "Calibration Curve (Reliability Diagram)"),
                ("logs/threshold_analysis.png",       "Threshold Analysis (Cost + F1 vs Threshold)"),
                ("logs/calibrated_confusion_matrix.png", "Confusion Matrix (Cost-Optimal Threshold)"),
                ("logs/fold_stability.png",           "Cross-Fold F1 Stability"),
                ("logs/segment_evaluation.png",       "Segment Evaluation (projects_completed)"),
            ]:
                if os.path.exists(img_path):
                    gr.Image(img_path, label=img_label)

        # ── Tab 6: Task 12 — How It Works ──────────────────────────
        with gr.Tab("📖 Task 12 — How It Works"):
            gr.Markdown(
                """
                ## Task 12 — Calibrated Binary Classification

                ### Why Calibrate?
                Raw classifier scores (probabilities) are often **overconfident or underconfident**.
                CalibratedClassifierCV fits an isotonic regression (or Platt sigmoid) on held-out
                fold probabilities so that *P̂ = 0.7* truly means "70% of these cases are positive".

                ### Calibration Verification
                The **reliability diagram** (calibration curve) plots *fraction of positives* vs
                *mean predicted probability*.  A model on the diagonal = perfectly calibrated.
                The **Brier score** (lower is better) quantifies calibration numerically.

                ### Cost-Optimal Threshold
                Rather than defaulting to 0.5, we sweep all thresholds and minimise:
                > `total cost = FP × cost_FP + FN × cost_FN`

                In placement prediction, missing a real placement (FN) is twice as costly as
                a false alarm (FP), so `cost_FN = 2.0` vs `cost_FP = 1.0`.

                ### Stability & Fairness
                | Check | What it detects |
                |-------|-----------------|
                | 5-fold StratifiedKFold | Fragile models with high F1 variance across splits |
                | Segment evaluation | Hidden failure on a subgroup (e.g., low project count) |

                ### Production Packaging
                The final artifact is `models/calibrated_pipeline.pkl` — a single
                `CalibratedClassifierCV` object wrapping the full preprocessor + GB pipeline.
                Load with `joblib.load()` and call `.predict_proba()` for calibrated scores.
                """
            )


if __name__ == "__main__":
    print("Launching Task 11 + 12 Verification App...")
    demo.launch(
        share=True,
        theme=gr.themes.Base(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate"),
    )
