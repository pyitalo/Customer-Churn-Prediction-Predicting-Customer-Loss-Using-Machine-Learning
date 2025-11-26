# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import json

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
MANIFEST = MODELS_DIR / "manifest.json"

app = FastAPI(title="Customer Churn Prediction API (Pipeline)")

def latest_model_path():
    with open(MANIFEST, "r") as f:
        data = json.load(f)
    last = data["models"][-1]
    return MODELS_DIR / last["path"]

pipeline = joblib.load(latest_model_path())

class CustomerIn(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    InternetService: str
    Contract: str
    PhoneService: str
    MultipleLines: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"status":"ok"}

@app.post("/predict")
def predict(payload: CustomerIn):
    import pandas as pd
    data = payload.dict()

    df = pd.DataFrame([data])

    pred = pipeline.predict(df)[0]
    prob = float(pipeline.predict_proba(df)[0][1])

    return {"churn": int(pred), "probability": round(prob, 4)}


