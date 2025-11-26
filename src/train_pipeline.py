# src/train_pipeline.py
import os
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MANIFEST = MODELS_DIR / "manifest.json"

def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)

def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])
    return pipeline

def register_model(model_path, metadata):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text())
    else:
        data = {"models": []}
    data["models"].append(metadata)
    MANIFEST.write_text(json.dumps(data, indent=2))

def train_and_save():
    df = load_dataset()
    # safe drop if exists
    df = df.drop(columns=["customerID"], errors="ignore")
    # ensure numeric conversions
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # simple fill
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    # target
    df["Churn"] = df["Churn"].astype(str)
    # features: use a richer set (improves performance vs logistic simple)
    # keep columns that exist
    candidate_features = ["gender","SeniorCitizen","Partner","Dependents","tenure",
                          "InternetService","Contract","PhoneService","MultipleLines",
                          "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
                          "StreamingTV","StreamingMovies","PaperlessBilling","PaymentMethod",
                          "MonthlyCharges","TotalCharges"]
    features = [c for c in candidate_features if c in df.columns]
    # Separate X/y
    X = df[features]
    y = (df["Churn"].astype(str).str.lower() == "yes").astype(int)
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # define numeric / categorical
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    pipeline = build_pipeline(numeric_features, categorical_features)
    # fit
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=3)
    print("Accuracy:", acc)
    print("Report:\n", report)
    # versioning: timestamp
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_filename = f"pipeline_v{ts}.pkl"
    model_path = MODELS_DIR / model_filename
    joblib.dump(pipeline, model_path)
    print("Saved model:", model_path)
    # register
    metadata = {
        "version": ts,
        "path": str(model_path.name),
        "accuracy": float(acc),
        "trained_at": ts,
        "features": features
    }
    register_model(model_path, metadata)
    print("Model registered in manifest.")
    return model_path, metadata

if __name__ == "__main__":
    train_and_save()
