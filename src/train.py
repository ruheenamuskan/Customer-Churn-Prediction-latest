# src/train.py
"""
Train script for Telco Customer Churn.
Builds an end-to-end scikit-learn pipeline (Imputation, Scaling, One-Hot Encoding, Classifier)
and saves the trained pipeline to ../models/churn_pipeline.joblib
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
)

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(ROOT, "models", "churn_pipeline.joblib")

# Column Definitions
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

def load_and_clean(path):
    """Load raw dataset and perform initial data type corrections and sanitization."""
    df = pd.read_csv(path)
    
    # Drop customerID identifier if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    # Convert TotalCharges to numeric (handles empty spaces/strings gracefully)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Impute missing TotalCharges with 0.0 (typically tenure=0 customers)
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        
    # Ensure SeniorCitizen is numeric
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        
    # Drop rows without target
    if "Churn" in df.columns:
        df = df.dropna(subset=["Churn"])
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        
    return df

def build_pipeline():
    """Builds a scikit-learn ColumnTransformer and Classifier Pipeline."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES)
    ], remainder="drop")

    # Tuned Random Forest Classifier with balanced weights for churn classification
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

    return pipeline

def main():
    print("=" * 60)
    print(">>> Training Telco Customer Churn Prediction Pipeline")
    print("=" * 60)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        
    df = load_and_clean(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True).round(4) * 100}%\n")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    pipeline = build_pipeline()

    # Stratified Train-Test Split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    print("\n--- Evaluation on Test Split (20%) ---")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"PR AUC:    {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save trained pipeline
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[OK] Pipeline successfully saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()


