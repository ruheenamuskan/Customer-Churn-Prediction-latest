# src/evaluate.py
"""
Load saved pipeline and run comprehensive evaluation on test split from dataset.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(ROOT, "models", "churn_pipeline.joblib")

def load_and_clean(path):
    """Load and sanitize the dataset."""
    df = pd.read_csv(path)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    if "Churn" in df.columns:
        df = df.dropna(subset=["Churn"])
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def main():
    print("=" * 60)
    print(">>> Evaluating Saved Churn Prediction Pipeline")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file not found at: {MODEL_PATH}")
        print("Please train the model first by running: python src/train.py")
        return

    if not os.path.exists(DATA_PATH):
        print(f"[Error] Dataset not found at: {DATA_PATH}")
        return

    df = load_and_clean(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"Evaluated on: {len(y_test)} test samples")
    print(f"Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC:      {roc_auc:.4f}")
    print(f"PR-AUC:       {pr_auc:.4f}\n")
    
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn (0)", "Churn (1)"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print("--- Confusion Matrix ---")
    print(f"True Negatives:  {cm[0][0]} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]} | True Positives:  {cm[1][1]}")

if __name__ == "__main__":
    main()


