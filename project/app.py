import gradio as gr
import pandas as pd
import numpy as np
import json
import os
import sys

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

    with open(path) as f:
        m = json.load(f)

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

with gr.Blocks(title="PlaceMux Ensemble — Task 11") as demo:

    gr.Markdown(
        """
        # 🧠 PlaceMux Placement Predictor — Task 11: Ensemble Learning
        **Live verification UI.** Powered by a Voting + Stacking ensemble of three diverse models
        (Logistic Regression · Random Forest · Gradient Boosting).
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


if __name__ == "__main__":
    print("Launching Task 11 Ensemble Verification App...")
    demo.launch(
        share=True,
        theme=gr.themes.Base(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate"),
    )
