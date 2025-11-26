# src/predict.py
from pathlib import Path
import joblib
import pandas as pd
import json

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
MANIFEST = MODELS_DIR / "manifest.json"

def latest_model_path():
    if not MANIFEST.exists():
        raise FileNotFoundError("Manifest not found — treine o modelo primeiro.")
    with open(MANIFEST, "r") as f:
        data = json.load(f)
    last = data["models"][-1]
    return MODELS_DIR / last["path"]

def load_pipeline():
    path = latest_model_path()
    return joblib.load(path)

def make_prediction(pipeline, input_dict):
    df = pd.DataFrame([input_dict])
    # pipeline will handle feature selection/encoding
    preds = pipeline.predict(df)
    probs = pipeline.predict_proba(df)[:,1]
    return int(preds[0]), float(probs[0])

if __name__ == "__main__":
    pipeline = load_pipeline()
    sample = {
        "gender":"Male",
        "SeniorCitizen":0,
        "Partner":"Yes",
        "Dependents":"No",
        "tenure":12,
        "InternetService":"Fiber optic",
        "Contract":"Month-to-month",
        "PhoneService":"Yes",
        "MultipleLines":"No",
        "OnlineSecurity":"No",
        "OnlineBackup":"No",
        "DeviceProtection":"No",
        "TechSupport":"No",
        "StreamingTV":"No",
        "StreamingMovies":"No",
        "PaperlessBilling":"Yes",
        "PaymentMethod":"Electronic check",
        "MonthlyCharges":75.5,
        "TotalCharges":350.4
    }
    pred, prob = make_prediction(pipeline, sample)
    print("Pred:", pred, "Prob:", prob)
