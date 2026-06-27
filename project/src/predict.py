import pandas as pd
import numpy as np
import joblib
import sys
import os

from src.data import feature_engineering

# ---------------------------------------------------------
# DEPENDENCY & FAILURE HANDLING
# ---------------------------------------------------------
def load_artifacts():
    """Load model pipeline with robust error handling."""
    pipeline_path = "models/pipeline.pkl"
    
    if not os.path.exists(pipeline_path):
        print(f"[ERROR] Missing model pipeline. Please run `python -m src.train` first.")
        sys.exit(1)
        
    try:
        pipeline = joblib.load(pipeline_path)
        return pipeline
    except Exception as e:
        print(f"[ERROR] Failed to load pipeline: {str(e)}")
        sys.exit(1)


def predict_live(candidate_data):
    """
    Live verification function.
    Demonstrates edge-case handling for missing values and wrong types.
    """
    print(f"\n--- LIVE VERIFICATION ---")
    print(f"Input Data: {candidate_data}")
    
    # EDGE CASE: Empty input
    if not candidate_data:
        print("[ERROR] No candidate data provided.")
        return None
        
    try:
        # 1. Convert to DataFrame (handles schema alignment)
        df_raw = pd.DataFrame([candidate_data])
        
        # Define all expected base columns that feature_engineering expects
        expected_cols = [
            'registration_date', 'last_login_date', 'domain_score', 
            'aptitude_score', 'projects_completed', 'has_offer_letter', 
            'days_to_placement', 'random_noise'
        ]
        
        # EDGE CASE: Fill missing columns with NaN to prevent KeyErrors
        for col in expected_cols:
            if col not in df_raw.columns:
                df_raw[col] = np.nan
        
        # EDGE CASE: Ensure required date columns exist for feature engineering
        # If missing or NaN, we simulate them
        if pd.isna(df_raw['registration_date'].iloc[0]) or pd.isna(df_raw['last_login_date'].iloc[0]):
            print("[WARNING] Missing date fields. Defaulting to 0 active days.")
            df_raw['registration_date'] = pd.to_datetime('today')
            df_raw['last_login_date'] = pd.to_datetime('today')
        else:
            # Type conversion error handling
            df_raw['registration_date'] = pd.to_datetime(df_raw['registration_date'])
            df_raw['last_login_date'] = pd.to_datetime(df_raw['last_login_date'])
            
        # 2. Apply Feature Engineering (derive candidates, prune leaks)
        # Note: prune_leaks=True drops dates, which is required for the pipeline
        df_engineered = feature_engineering(df_raw, prune_leaks=True)
        
        # Remove target variable if it was accidentally provided
        if 'placed' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['placed'])
            
        # 3. Load Artifacts
        pipeline = load_artifacts()
        
        # 4. Predict (Pipeline automatically applies preprocessing)
        prediction = pipeline.predict(df_engineered)[0]
        probability = pipeline.predict_proba(df_engineered)[0][1] if hasattr(pipeline, 'predict_proba') else None
        
        result = "Placed" if prediction == 1 else "Not Placed"
        print(f"\n[SUCCESS] Prediction: {result}")
        if probability is not None:
            print(f"Probability of Placement: {probability:.2%}")
            
        return prediction
        
    except Exception as e:
        print(f"\n[ERROR] Live Prediction Failed during processing:")
        print(f"Details: {str(e)}")
        return None


if __name__ == "__main__":
    # Test Case 1: Normal Valid Candidate
    sample_candidate = {
        "registration_date": "2026-01-15",
        "last_login_date": "2026-03-20",
        "domain_score": 85.5,
        "aptitude_score": 90.0,
        "projects_completed": 4,
        # Leaky features that will be automatically pruned safely:
        "has_offer_letter": 1, 
        "days_to_placement": 15
    }
    
    predict_live(sample_candidate)
    
    # Test Case 2: Edge Case - Missing numeric fields (handled by Imputer)
    edge_case_candidate = {
        "registration_date": "2026-05-01",
        "last_login_date": "2026-05-10",
        # Missing domain_score and aptitude_score
        "projects_completed": 1
    }
    predict_live(edge_case_candidate)
    
    # Test Case 3: Edge Case - Missing date fields
    missing_dates_candidate = {
        "domain_score": 45,
        "aptitude_score": 50,
        "projects_completed": 0
    }
    predict_live(missing_dates_candidate)
