# src/predict_api.py
"""
Flask REST API for Telco Customer Churn Prediction.
Accepts POST requests at /predict with JSON payload and returns prediction, probability,
risk level, and key risk indicators.
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "..", "models", "churn_pipeline.joblib")
_model = None

# Expected features and sensible defaults
DEFAULT_PAYLOAD = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 50.0,
    "TotalCharges": 50.0,
}

def load_model():
    """Lazy-load the serialized machine learning pipeline."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def sanitize_input(data: dict) -> pd.DataFrame:
    """Validate, fill missing defaults, and enforce numeric data types."""
    merged = {**DEFAULT_PAYLOAD, **data}
    
    # Cast numerical values
    merged["tenure"] = int(float(merged.get("tenure", 0)))
    merged["SeniorCitizen"] = int(float(merged.get("SeniorCitizen", 0)))
    merged["MonthlyCharges"] = float(merged.get("MonthlyCharges", 0.0))
    merged["TotalCharges"] = float(merged.get("TotalCharges", 0.0))
    
    return pd.DataFrame([merged])

def extract_risk_factors(customer: dict, proba: float) -> list:
    """Identify top intuitive risk indicators for business retention context."""
    factors = []
    if customer.get("Contract") == "Month-to-month":
        factors.append("Month-to-month contract (high cancellation flexibility)")
    if float(customer.get("tenure", 0)) < 12:
        factors.append(f"Short tenure ({customer.get('tenure')} months) - high early lifecycle risk")
    if float(customer.get("MonthlyCharges", 0)) > 70:
        factors.append(f"High monthly bill (${customer.get('MonthlyCharges'):.2f}/month)")
    if customer.get("InternetService") == "Fiber optic" and customer.get("TechSupport") == "No":
        factors.append("Fiber optic subscription without technical support")
    if customer.get("OnlineSecurity") == "No":
        factors.append("No online security add-on")
    if customer.get("PaymentMethod") == "Electronic check":
        factors.append("Payment via Electronic Check (statistically higher churn rate)")
    return factors

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": "Telco Customer Churn Prediction API",
        "endpoints": {
            "/predict": "POST - Predict churn risk for a customer JSON payload",
            "/health": "GET - Check API and model readiness"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    try:
        load_model()
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON:
    {
      "gender": "Female",
      "SeniorCitizen": 0,
      "tenure": 3,
      "Contract": "Month-to-month",
      "MonthlyCharges": 85.5,
      ...
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a valid JSON object"}), 400

        pipeline = load_model()
        df = sanitize_input(data)

        # Generate probabilities and prediction
        proba = float(pipeline.predict_proba(df)[:, 1][0])
        pred = int(pipeline.predict(df)[0])

        # Categorize risk level
        if proba >= 0.65:
            risk_level = "High"
        elif proba >= 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        risk_factors = extract_risk_factors(df.iloc[0].to_dict(), proba)

        return jsonify({
            "churn": pred,
            "probability": round(proba, 4),
            "risk_level": risk_level,
            "risk_factors": risk_factors
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Pre-loading model...")
    try:
        load_model()
        print("Model loaded successfully.")
    except Exception as err:
        print(f"Warning: {err}")
        
    app.run(host="0.0.0.0", port=5000, debug=True)

