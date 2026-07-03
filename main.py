import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Loan Default Prediction API")

# Setup CORS to allow Next.js frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Attempt to load model
MODEL_PATH = "loan_default_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {MODEL_PATH}. Using fallback mock prediction. Error: {e}")

class PredictionRequest(BaseModel):
    # These should match the expected inputs
    # For now we use generic dictionary input
    features: dict

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    explanation: str
    status: str

@app.get("/")
def read_root():
    return {"status": "Model API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        # Fallback prediction for testing the UI
        import random
        prob = random.uniform(0.1, 0.9)
        pred = 1 if prob > 0.5 else 0
        risk_level = "High" if pred == 1 else "Low" if prob < 0.3 else "Medium"
        return {
            "prediction": pred,
            "probability": prob,
            "risk_level": risk_level,
            "explanation": "This is a simulated prediction. The model could not be loaded.",
            "status": "mock"
        }
    
    try:
        # Convert incoming JSON dict to DataFrame
        df = pd.DataFrame([request.features])
        
        # Here we would normally apply the exact preprocessing steps
        # such as one-hot encoding, scaling, etc.
        # df = preprocess(df)
        
        # For this skeleton, we assume df matches model's expected features
        # Or that it will fail and we catch the exception.
        proba = model.predict_proba(df)[0, 1]
        pred = int(proba >= 0.5)
        
        risk_level = "High" if pred == 1 else "Low" if proba < 0.3 else "Medium"
        
        return {
            "prediction": pred,
            "probability": float(proba),
            "risk_level": risk_level,
            "explanation": f"The AI predicts a {risk_level.lower()} risk of default with {proba:.1%} probability based on the provided profile.",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
