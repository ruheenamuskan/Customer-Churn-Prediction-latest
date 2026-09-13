import os
import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 📂 Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "Telco-Customer-Churn.csv")
MODEL_PIPELINE_PATH = os.path.join(ROOT, "models", "churn_pipeline.joblib")
MODEL_PKL_PATH = os.path.join(ROOT, "model.pkl")

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

st.set_page_config(page_title="Customer Churn Analytics & Prediction", layout="wide", page_icon="🔮")

# --- Load dataset ---
@st.cache_data
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0).astype(float)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(1).astype(int)
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(50.0).astype(float)
    return df

DEFAULT_COLUMNS = {
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

def sanitize_custom_df(df_input: pd.DataFrame) -> pd.DataFrame:
    df_clean = df_input.copy()
    for col, default_val in DEFAULT_COLUMNS.items():
        if col not in df_clean.columns:
            df_clean[col] = default_val
    for col in NUMERIC_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0).astype(float)
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
    return df_clean

# --- Build Pipeline ---
def build_pipeline():
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

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    return Pipeline([("preprocessor", preprocessor), ("clf", clf)])

# --- Train & Cache Model ---
def fit_pipeline_on_data(df: pd.DataFrame):
    df_clean = df.copy()
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])
    if "Churn" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["Churn"])
        df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    
    df_clean = sanitize_custom_df(df_clean)
    X = df_clean.drop(columns=["Churn"], errors="ignore")
    y = df_clean["Churn"] if "Churn" in df_clean.columns else None

    if y is not None:
        pipeline = build_pipeline()
        pipeline.fit(X, y)
        os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
        try:
            joblib.dump(pipeline, MODEL_PIPELINE_PATH)
        except Exception:
            pass
        return pipeline
    return None

@st.cache_resource
def load_model():
    """Load pre-trained model or fit on the fly on Streamlit Cloud."""
    if os.path.exists(MODEL_PIPELINE_PATH):
        try:
            m = joblib.load(MODEL_PIPELINE_PATH)
            # Validate model compatibility against current sklearn environment
            test_row = sanitize_custom_df(pd.DataFrame([DEFAULT_COLUMNS]))
            m.predict_proba(test_row)
            return m
        except Exception:
            pass
    df_inbuilt = load_data()
    if not df_inbuilt.empty and "Churn" in df_inbuilt.columns:
        return fit_pipeline_on_data(df_inbuilt)
    return None

def train_model(df: pd.DataFrame):
    return fit_pipeline_on_data(df)

# --- Generate reason for churn ---
def churn_reason(row):
    reasons = []
    if str(row.get("Contract", "")) == "Month-to-month":
        reasons.append("Month-to-month contract")
    if float(row.get("tenure", 0)) < 12:
        reasons.append("Low tenure (< 12 mo)")
    if float(row.get("MonthlyCharges", 0)) > 75:
        reasons.append("High monthly bill")
    if str(row.get("TechSupport", "")) == "No" and str(row.get("InternetService", "")) != "No":
        reasons.append("No tech support")
    if str(row.get("PaymentMethod", "")) == "Electronic check":
        reasons.append("Electronic check payment")
    if not reasons:
        reasons.append("General churn risk factors")
    return ", ".join(reasons)


def get_active_dataset(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = sanitize_custom_df(df)
            return df, f"Custom Uploaded File: {uploaded_file.name}", True
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
            df_def = load_data()
            return df_def, "Default Inbuilt Dataset (Telco-Customer-Churn.csv)", False
    else:
        df_def = load_data()
        return df_def, "Default Inbuilt Dataset (Telco-Customer-Churn.csv)", False

# --- Streamlit App ---
st.title("🔮 Customer Churn Analytics & Batch Prediction")
st.markdown("Filter customer segments, evaluate churn probabilities, inspect revenue at risk, and export prediction datasets.")

with st.expander("📤 Upload Custom Customer CSV (or Download Sample Datasets)", expanded=False):
    st.markdown("##### 📥 Don't have a dataset? Download sample files directly:")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "rb") as f:
                st.download_button(
                    label="📥 Download Full Real 7,043 Dataset (CSV)",
                    data=f.read(),
                    file_name="Telco-Customer-Churn-Real-7043.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    with dl_col2:
        test_csv_path = os.path.join(ROOT, "sample_test_customers.csv")
        if os.path.exists(test_csv_path):
            with open(test_csv_path, "rb") as f:
                st.download_button(
                    label="📥 Download 50-Customer Test Dataset (CSV)",
                    data=f.read(),
                    file_name="sample_test_customers_50.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    st.divider()
    uploaded_file = st.file_uploader(
        "Upload your custom customer CSV file for batch churn prediction",
        type=["csv"],
        help="Upload your own dataset. If left blank, the built-in 7,043 customer dataset is used."
    )


df, data_source_label, is_custom_data = get_active_dataset(uploaded_file)
if df.empty:
    st.stop()

# Top action bar
top_col1, top_col2, top_col3 = st.columns([3, 1, 1])
with top_col1:
    if is_custom_data:
        st.success(f"📂 **Active Dataset:** {data_source_label} ({len(df):,} records loaded)")
    else:
        st.info(f"📂 **Active Dataset:** {data_source_label} ({len(df):,} records loaded)")
with top_col2:
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.session_state["app_gender"] = "Any"
        st.session_state["app_senior"] = "Any"
        st.session_state["app_contract"] = "Any"
        st.session_state["app_internet"] = "Any"
        st.session_state["app_tenure"] = (int(df["tenure"].min()), int(df["tenure"].max()))
        st.session_state["app_charges"] = (float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()))
        st.rerun()
with top_col3:
    if st.button("🛠️ Retrain Model Pipeline", use_container_width=True):
        with st.spinner("Training Random Forest Pipeline..."):
            trained_model = train_model(df)
            if trained_model:
                st.success("✅ Model retrained and saved!")
            else:
                st.error("❌ Training failed.")

model = load_model()
if model is None:
    st.warning("⚠️ No trained model found. Click 'Retrain Model Pipeline' above or run `python src/train.py`.")


st.divider()
st.subheader("🔍 Segment Filters")

fcol1, fcol2, fcol3, fcol4 = st.columns(4)

with fcol1:
    gender_opts = ["Any"] + sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else ["Any"]
    gender_filter = st.selectbox("Gender", gender_opts)

with fcol2:
    senior_opts = ["Any", "Yes (1)", "No (0)"] if "SeniorCitizen" in df.columns else ["Any"]
    senior_filter = st.selectbox("Senior Citizen", senior_opts)

with fcol3:
    contract_opts = ["Any"] + sorted(df["Contract"].dropna().unique().tolist()) if "Contract" in df.columns else ["Any"]
    contract_filter = st.selectbox("Contract Type", contract_opts)

with fcol4:
    internet_opts = ["Any"] + sorted(df["InternetService"].dropna().unique().tolist()) if "InternetService" in df.columns else ["Any"]
    internet_filter = st.selectbox("Internet Service", internet_opts)

# Numeric ranges
rcol1, rcol2 = st.columns(2)
with rcol1:
    min_t, max_t = int(df["tenure"].min()), int(df["tenure"].max())
    tenure_range = st.slider("Tenure Range (Months)", min_value=min_t, max_value=max_t, value=(min_t, max_t))

with rcol2:
    min_m, max_m = float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())
    charge_range = st.slider("Monthly Charges Range ($)", min_value=min_m, max_value=max_m, value=(min_m, max_m), format="$%.2f")

# Filter logic
matched = df.copy()
if gender_filter != "Any":
    matched = matched[matched["gender"] == gender_filter]
if senior_filter != "Any":
    s_val = 1 if "Yes" in senior_filter else 0
    matched = matched[matched["SeniorCitizen"] == s_val]
if contract_filter != "Any":
    matched = matched[matched["Contract"] == contract_filter]
if internet_filter != "Any":
    matched = matched[matched["InternetService"] == internet_filter]

matched = matched[
    (matched["tenure"] >= tenure_range[0]) & (matched["tenure"] <= tenure_range[1]) &
    (matched["MonthlyCharges"] >= charge_range[0]) & (matched["MonthlyCharges"] <= charge_range[1])
]

st.divider()

if st.button("🚀 Run Churn Prediction on Filtered Segment", type="primary", use_container_width=True):
    if model is None:
        st.error("Please train a model first.")
    elif matched.empty:
        st.warning("⚠️ No customers found matching the selected filters.")
    else:
        X = matched.copy()
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])
        if "Churn" in X.columns:
            X = X.drop(columns=["Churn"])
        X = sanitize_custom_df(X)

        # Predict using full pipeline
        probas = model.predict_proba(X)[:, 1]
        preds = model.predict(X)


        matched["Churn Probability"] = (probas * 100).round(1).astype(str) + "%"
        matched["Churn Prediction"] = preds
        matched["Risk Status"] = np.where(probas >= 0.65, "High Risk 🔴", np.where(probas >= 0.35, "Medium Risk 🟡", "Low Risk 🟢"))
        matched["Primary Churn Drivers"] = matched.apply(
            lambda row: churn_reason(row) if row["Churn Prediction"] == 1 else "Low Risk", axis=1
        )

        total_customers = len(matched)
        churn_count = int(np.sum(preds == 1))
        churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0
        revenue_at_risk = matched[matched["Churn Prediction"] == 1]["MonthlyCharges"].sum()

        # KPI Cards
        st.subheader("📊 Segment Performance & Risk Overview")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Matching Customers", f"{total_customers:,}")
        kpi2.metric("Predicted Churners", f"{churn_count:,}")
        kpi3.metric("Segment Churn Rate", f"{churn_rate:.1f}%")
        kpi4.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.2f}")

        # Visualizations
        st.markdown("### 📈 Visual Analytics")
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            st.markdown("#### Churn Risk Breakdown")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            risk_counts = matched["Risk Status"].value_counts()
            colors = {"High Risk 🔴": "#ff6666", "Medium Risk 🟡": "#ffcc66", "Low Risk 🟢": "#66b3ff"}
            slice_colors = [colors.get(k, "#cccccc") for k in risk_counts.index]
            ax1.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%", startangle=90, colors=slice_colors)
            ax1.axis("equal")
            st.pyplot(fig1)

        with vcol2:
            st.markdown("#### Top Churn Drivers")
            churners = matched[matched["Churn Prediction"] == 1]
            if not churners.empty:
                reason_counts = churners["Primary Churn Drivers"].value_counts().head(6)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.barplot(y=reason_counts.index, x=reason_counts.values, ax=ax2, palette="Reds_r")
                ax2.set_xlabel("Number of At-Risk Customers")
                ax2.set_ylabel("")
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("No churners predicted in this segment.")

        # Data Table & Download
        st.subheader("📋 Segment Predictions Table")
        st.dataframe(matched, use_container_width=True)

        csv_data = matched.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Segment Predictions (CSV)",
            data=csv_data,
            file_name="telco_churn_segment_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

