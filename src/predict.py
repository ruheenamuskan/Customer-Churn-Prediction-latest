import os
import sys
import json
import ast
import re
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "churn_pipeline.joblib")

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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python src/train.py' first.")
    return joblib.load(MODEL_PATH)

def sanitize_input(data_dict: dict) -> pd.DataFrame:
    merged = {**DEFAULT_PAYLOAD, **data_dict}
    merged["tenure"] = int(float(merged.get("tenure", 0)))
    merged["SeniorCitizen"] = int(float(merged.get("SeniorCitizen", 0)))
    merged["MonthlyCharges"] = float(merged.get("MonthlyCharges", 0.0))
    merged["TotalCharges"] = float(merged.get("TotalCharges", 0.0))
    return pd.DataFrame([merged])

def parse_input_payload(raw_str: str) -> dict:
    raw_str = raw_str.strip()
    if (raw_str.startswith("'") and raw_str.endswith("'")) or (raw_str.startswith('"') and raw_str.endswith('"')):
        raw_str = raw_str[1:-1].strip()

    # 1. Direct JSON
    try:
        return json.loads(raw_str)
    except Exception:
        pass

    # 2. Python Dict literal
    try:
        res = ast.literal_eval(raw_str)
        if isinstance(res, dict):
            return res
    except Exception:
        pass

    # 3. Robust key-value pair parser (handles spaces, parentheses, unquoted text)
    try:
        content = raw_str.strip("{}")
        parsed = {}
        # Split by commas that separate top-level key-values
        items = [item.strip() for item in content.split(",") if item.strip()]
        for item in items:
            if ":" in item:
                k, v = item.split(":", 1)
                k = k.strip().strip("'\"")
                v = v.strip().strip("'\"")
                # Try numeric conversion
                if v.isdigit():
                    v = int(v)
                else:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                parsed[k] = v
        if parsed:
            return parsed
    except Exception:
        pass

    raise ValueError(f"Could not parse input string into a dictionary: {raw_str}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py '<json_string>'")
        print("Example:\n  python src/predict.py '{\"Contract\":\"Month-to-month\",\"tenure\":3,\"MonthlyCharges\":85.5}'")
        sys.exit(1)

    raw_input = " ".join(sys.argv[1:])
    try:
        data = parse_input_payload(raw_input)
    except Exception as err:
        print(f"[Error] Invalid input format - {err}")
        sys.exit(1)



    try:
        pipeline = load_model()
        df = sanitize_input(data)

        proba = float(pipeline.predict_proba(df)[:, 1][0])
        pred = int(pipeline.predict(df)[0])
        
        risk_level = "HIGH" if proba >= 0.65 else ("MEDIUM" if proba >= 0.35 else "LOW")
        status = "CHURN (Will Leave)" if pred == 1 else "NO CHURN (Will Stay)"

        print("\n" + "=" * 45)
        print(">>> Customer Churn Prediction Result")
        print("=" * 45)
        print(f"  Prediction:          {status}")
        print(f"  Churn Probability:   {proba * 100:.2f}%")
        print(f"  Risk Level:          {risk_level}")
        print("=" * 45 + "\n")

    except Exception as e:
        print(f"[Error] Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


