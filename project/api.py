import time
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

# Ensure src is in the path or just import it since api.py is in project root
try:
    from src.data import feature_engineering
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.data import feature_engineering

# --- Schemas ---

class CandidateFeatureInput(BaseModel):
    registration_date: str = Field(..., example="2026-01-15")
    last_login_date: str = Field(..., example="2026-03-20")
    domain_score: float = Field(..., example=85.5)
    aptitude_score: float = Field(..., example=90.0)
    projects_completed: int = Field(..., example=4)

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    is_placed: bool
    threshold: float
    version: str

class HealthOutput(BaseModel):
    status: str
    version: str
    model_loaded: bool

# --- App & Router setup ---

app = FastAPI(
    title="PlaceMux Model Serving API",
    description="Live REST API for the PlaceMux Phase 1 Industry Immersion model serving (Task 13).",
    version="1.0.0"
)

router_v1 = APIRouter(prefix="/v1")

# Global state for model
MODEL_PATH = "models/calibrated_pipeline.pkl"
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    if os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Prediction endpoint will fail.")

# --- Middleware ---

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware for latency monitoring.
    Calculates request processing time and adds it to the X-Process-Time header.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # We could also log this to a monitoring system
    # print(f"Method: {request.method} Path: {request.url.path} Latency: {process_time:.4f}s")
    
    return response

# --- Endpoints ---

@router_v1.get("/health", response_model=HealthOutput)
def health_check():
    """
    Health check endpoint for deployment monitoring.
    """
    return {
        "status": "healthy",
        "version": app.version,
        "model_loaded": pipeline is not None
    }

@router_v1.post("/predict", response_model=PredictionOutput)
def predict(input_data: CandidateFeatureInput, threshold: float = 0.40):
    """
    Live model prediction endpoint. Takes candidate details and returns calibrated probability.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train the model first.")

    try:
        # Reconstruct DataFrame as expected by the pipeline
        df_raw = pd.DataFrame([{
            "registration_date": input_data.registration_date,
            "last_login_date":   input_data.last_login_date,
            "domain_score":      input_data.domain_score,
            "aptitude_score":    input_data.aptitude_score,
            "projects_completed": input_data.projects_completed,
            "has_offer_letter":  np.nan,
            "days_to_placement": np.nan,
            "random_noise":      np.nan,
        }])
        
        df_raw["registration_date"] = pd.to_datetime(df_raw["registration_date"])
        df_raw["last_login_date"]   = pd.to_datetime(df_raw["last_login_date"])

        df_eng = feature_engineering(df_raw, prune_leaks=True)
        if "placed" in df_eng.columns:
            df_eng = df_eng.drop(columns=["placed"])

        # Predict
        prob = pipeline.predict_proba(df_eng)[0][1]
        pred = int(prob >= threshold)

        return {
            "prediction": pred,
            "probability": prob,
            "is_placed": pred == 1,
            "threshold": threshold,
            "version": app.version
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Register routers
app.include_router(router_v1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
