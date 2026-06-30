import gradio as gr
import pandas as pd
import numpy as np
import sys

from src.predict import load_artifacts
from src.data import feature_engineering

def predict_candidate(reg_date, login_date, domain, aptitude, projects):
    try:
        pipeline = load_artifacts()
        
        df_raw = pd.DataFrame([{
            "registration_date": reg_date,
            "last_login_date": login_date,
            "domain_score": domain,
            "aptitude_score": aptitude,
            "projects_completed": projects,
            "has_offer_letter": np.nan,
            "days_to_placement": np.nan,
            "random_noise": np.nan
        }])
        
        df_raw['registration_date'] = pd.to_datetime(df_raw['registration_date'])
        df_raw['last_login_date'] = pd.to_datetime(df_raw['last_login_date'])
        
        df_engineered = feature_engineering(df_raw, prune_leaks=True)
        if 'placed' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['placed'])
            
        pred = pipeline.predict(df_engineered)[0]
        prob = pipeline.predict_proba(df_engineered)[0][1] if hasattr(pipeline, 'predict_proba') else None
        
        result = "✅ Placed" if pred == 1 else "❌ Not Placed"
        if prob is not None:
            result += f" (Probability: {prob:.2%})"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="PlaceMux Predictor") as demo:
    gr.Markdown("# 🚀 PlaceMux Placement Prediction Model (Task 10)")
    gr.Markdown("Live verification UI. Enter candidate data below to test the trained Gradient Boosting model.")
    
    with gr.Row():
        reg_date = gr.Textbox(label="Registration Date (YYYY-MM-DD)", value="2026-01-15")
        login_date = gr.Textbox(label="Last Login Date (YYYY-MM-DD)", value="2026-03-20")
    
    with gr.Row():
        domain = gr.Number(label="Domain Score", value=85.5)
        aptitude = gr.Number(label="Aptitude Score", value=90.0)
        projects = gr.Number(label="Projects Completed", value=4)
        
    btn = gr.Button("Predict Placement", variant="primary")
    output = gr.Textbox(label="Prediction Result", text_align="center")
    
    btn.click(predict_candidate, inputs=[reg_date, login_date, domain, aptitude, projects], outputs=output)

if __name__ == "__main__":
    print("Launching Live Verification App...")
    # share=True creates a public gradio.live URL
    demo.launch(share=True)
