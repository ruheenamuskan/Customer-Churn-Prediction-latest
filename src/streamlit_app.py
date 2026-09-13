# src/streamlit_app.py
"""
Unified Customer Churn Analytics, Segment Explorer, & Retention Intelligence Dashboard.
Features:
- Tab 1: 📊 Segment Explorer & Visual Analytics (Filters, KPI Cards, Pie Chart, Bar Chart, Customer Table, CSV Export)
- Tab 2: 🔮 Single Customer Prediction & Retention Recommendations
"""

import os
import requests
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "..", "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(ROOT, "..", "models", "churn_pipeline.joblib")
API_URL = "http://localhost:5000/predict"

st.set_page_config(
    page_title="Customer Churn Intelligence Hub",
    page_icon="🔮",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    return df

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# --- Feature defaults for custom uploaded datasets ---
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

def sanitize_custom_df(df_input):
    """Ensure all expected feature columns exist and types are aligned."""
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

def build_and_train_pipeline(df):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier

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

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

    df_clean = df.copy()
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])
    if "Churn" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["Churn"])
        df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    
    df_clean = sanitize_custom_df(df_clean)
    X = df_clean.drop(columns=["Churn"], errors="ignore")
    y = df_clean["Churn"]
    pipe.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        joblib.dump(pipe, MODEL_PATH)
    except Exception:
        pass
    return pipe

# --- Load Pipeline ---
@st.cache_resource
def load_pipeline():
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            test_row = sanitize_custom_df(pd.DataFrame([DEFAULT_COLUMNS]))
            m.predict_proba(test_row)
            return m
        except Exception:
            pass
    df_data = load_data()
    if not df_data.empty and "Churn" in df_data.columns:
        return build_and_train_pipeline(df_data)
    return None


def churn_reason(row):
    reasons = []
    if row.get("Contract") == "Month-to-month":
        reasons.append("Month-to-month contract")
    if float(row.get("tenure", 0)) < 12:
        reasons.append("Low tenure (<12 mo)")
    if float(row.get("MonthlyCharges", 0)) > 75:
        reasons.append("High monthly charges (>$75)")
    if row.get("TechSupport") == "No" and row.get("InternetService") != "No":
        reasons.append("No tech support")
    if row.get("PaymentMethod") == "Electronic check":
        reasons.append("Electronic check payment")
    if not reasons:
        reasons.append("General risk indicators")
    return ", ".join(reasons)

def get_retention_advice(customer_data, proba):
    advice = []
    if proba > 0.4:
        if customer_data["Contract"] == "Month-to-month":
            advice.append("🎯 **Contract Upgrade:** Offer a 1-year contract with an exclusive 15% discount for the first 3 months.")
        if customer_data["MonthlyCharges"] > 70:
            advice.append("💰 **Bill Optimization:** Review subscription bundle to offer a loyalty discount or family plan.")
        if customer_data["TechSupport"] == "No" and customer_data["InternetService"] != "No":
            advice.append("🛠️ **Support Assurance:** Grant 3 months of complimentary Priority Tech Support.")
        if customer_data["PaymentMethod"] == "Electronic check":
            advice.append("💳 **Automated Payments:** Offer a $5 bill credit for switching to Auto-Pay (Bank/Credit Card).")
        if customer_data["tenure"] <= 6:
            advice.append("📞 **Proactive Onboarding:** Schedule a 30-day satisfaction call from a customer success manager.")
    else:
        advice.append("✅ Customer exhibits low churn risk. Maintain standard relationship nurturing.")
    return advice

def get_active_dataset(uploaded_file):
    """Loads uploaded CSV if provided, else falls back to default inbuilt dataset."""
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

pipeline = load_pipeline()

st.title("🔮 Customer Churn Intelligence Hub")
st.markdown("Monitor customer segments, analyze visual churn trends (Pie & Bar charts), and simulate individual customer retention strategies.")

# --- Custom CSV Uploader in Expander / Header ---
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
        test_csv_path = os.path.join(ROOT, "..", "sample_test_customers.csv")
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
        "Upload your customer CSV file for batch churn prediction & analytics",
        type=["csv"],
        help="Upload your own dataset. If left blank, the app will automatically use the built-in 7,043-customer dataset."
    )


df_raw, data_source_label, is_custom_data = get_active_dataset(uploaded_file)

tab1, tab2 = st.tabs(["📊 Customer Segments & Visual Analytics", "👤 Single Customer Prediction & Retention"])

# ==========================================
# TAB 1: Segment Explorer & Visual Analytics
# ==========================================
with tab1:
    if df_raw.empty:
        st.error(f"Dataset not found at {DATA_PATH}")
    else:
        # Header with record count, data source indicator, and Reset button
        head_col1, head_col2 = st.columns([4, 1])
        with head_col1:
            if is_custom_data:
                st.success(f"📂 **Active Dataset:** {data_source_label} ({len(df_raw):,} records loaded)")
            else:
                st.info(f"📂 **Active Dataset:** {data_source_label} ({len(df_raw):,} records loaded)")
        with head_col2:
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state["tab1_gender"] = "Any"
                st.session_state["tab1_senior"] = "Any"
                st.session_state["tab1_contract"] = "Any"
                st.session_state["tab1_internet"] = "Any"
                st.session_state["tab1_tenure"] = (int(df_raw["tenure"].min()), int(df_raw["tenure"].max()))
                st.session_state["tab1_charges"] = (float(df_raw["MonthlyCharges"].min()), float(df_raw["MonthlyCharges"].max()))
                st.rerun()

        st.subheader("🔍 Segment Filters")
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)

        with fcol1:
            gender_opts = ["Any"] + sorted(df_raw["gender"].dropna().unique().tolist()) if "gender" in df_raw.columns else ["Any"]
            gender_filter = st.selectbox("Gender Filter", gender_opts, key="tab1_gender")

        with fcol2:
            senior_opts = ["Any", "Yes (1)", "No (0)"] if "SeniorCitizen" in df_raw.columns else ["Any"]
            senior_filter = st.selectbox("Senior Citizen Filter", senior_opts, key="tab1_senior")

        with fcol3:
            contract_opts = ["Any"] + sorted(df_raw["Contract"].dropna().unique().tolist()) if "Contract" in df_raw.columns else ["Any"]
            contract_filter = st.selectbox("Contract Filter", contract_opts, key="tab1_contract")

        with fcol4:
            internet_opts = ["Any"] + sorted(df_raw["InternetService"].dropna().unique().tolist()) if "InternetService" in df_raw.columns else ["Any"]
            internet_filter = st.selectbox("Internet Service Filter", internet_opts, key="tab1_internet")

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            min_t, max_t = int(df_raw["tenure"].min()), int(df_raw["tenure"].max())
            tenure_range = st.slider("Tenure (Months)", min_value=min_t, max_value=max_t, value=(min_t, max_t), key="tab1_tenure")

        with rcol2:
            min_m, max_m = float(df_raw["MonthlyCharges"].min()), float(df_raw["MonthlyCharges"].max())
            charge_range = st.slider("Monthly Charges ($)", min_value=min_m, max_value=max_m, value=(min_m, max_m), format="$%.2f", key="tab1_charges")

        # Apply Filters
        matched = df_raw.copy()
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


        if matched.empty:
            st.warning("⚠️ No customers match the selected filter criteria.")
            st.markdown(
                """
                > **Why is this happening?**
                > The combination of filters currently chosen has 0 matching customers. For example:
                > - Choosing **Internet Service = 'No'** with **Monthly Charges > $50** (landline-only customers typically pay under $30).
                > - Setting a **Tenure range** or **Monthly Charges slider** narrower than what exists for that specific filter.
                > 
                > 💡 **Solution:** Click the **'🔄 Reset Filters'** button above or broaden your slider ranges.
                """
            )

        else:
            if pipeline is None:
                st.error("❌ Trained pipeline not found. Run `python src/train.py` first.")
            else:
                X_seg = matched.copy()
                if "customerID" in X_seg.columns:
                    X_seg = X_seg.drop(columns=["customerID"])
                if "Churn" in X_seg.columns:
                    X_seg = X_seg.drop(columns=["Churn"])
                X_seg = sanitize_custom_df(X_seg)

                probas = pipeline.predict_proba(X_seg)[:, 1]
                preds = pipeline.predict(X_seg)


                matched["Churn Probability"] = (probas * 100).round(1).astype(str) + "%"
                matched["Churn Prediction"] = np.where(preds == 1, "Churn (1)", "No Churn (0)")
                matched["Risk Level"] = np.where(probas >= 0.65, "High Risk 🔴", np.where(probas >= 0.35, "Medium Risk 🟡", "Low Risk 🟢"))
                matched["Key Churn Reasons"] = matched.apply(
                    lambda row: churn_reason(row) if row["Churn Prediction"] == "Churn (1)" else "Low Risk", axis=1
                )

                total_cust = len(matched)
                churn_count = int(np.sum(preds == 1))
                churn_rate = (churn_count / total_cust * 100) if total_cust > 0 else 0
                rev_at_risk = matched[matched["Churn Prediction"] == "Churn (1)"]["MonthlyCharges"].sum()

                # KPI Cards
                st.divider()
                st.subheader("📈 Segment Overview & Financial Impact")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Matching Customers", f"{total_cust:,}")
                k2.metric("Predicted Churners", f"{churn_count:,}")
                k3.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
                k4.metric("Monthly Revenue at Risk", f"${rev_at_risk:,.2f}")

                # Visual Charts: Pie Chart and Bar Chart
                st.divider()
                st.subheader("📊 Visual Analytics: Churn Distribution & Key Reasons")
                vcol1, vcol2 = st.columns(2)

                with vcol1:
                    st.markdown("#### 🥧 Churn Distribution (Pie Chart)")
                    churn_counts = matched["Churn Prediction"].value_counts()
                    fig1, ax1 = plt.subplots(figsize=(4, 4))
                    colors = ["#66b3ff", "#ff6666"] if "No Churn (0)" in churn_counts.index else ["#ff6666"]
                    ax1.pie(
                        churn_counts,
                        labels=churn_counts.index,
                        autopct="%1.1f%%",
                        startangle=90,
                        colors=colors,
                        wedgeprops={"edgecolor": "white", "linewidth": 2}
                    )
                    ax1.axis("equal")
                    plt.tight_layout()
                    st.pyplot(fig1)

                with vcol2:
                    st.markdown("#### 📊 Top Churn Reasons (Bar Chart)")
                    churners_df = matched[matched["Churn Prediction"] == "Churn (1)"]
                    if not churners_df.empty:
                        # Extract all split reasons
                        all_reasons = []
                        for r_str in churners_df["Key Churn Reasons"]:
                            for r in r_str.split(", "):
                                if r and r != "Low Risk":
                                    all_reasons.append(r)
                        
                        if all_reasons:
                            reason_series = pd.Series(all_reasons).value_counts().head(6)
                            fig2, ax2 = plt.subplots(figsize=(6, 4))
                            sns.barplot(
                                y=reason_series.index,
                                x=reason_series.values,
                                ax=ax2,
                                palette="Reds_r"
                            )
                            ax2.set_xlabel("Number of At-Risk Customers")
                            ax2.set_ylabel("")
                            plt.tight_layout()
                            st.pyplot(fig2)
                        else:
                            st.info("No specific churn reasons flagged.")
                    else:
                        st.info("No churners found in this filtered segment.")

                # Detailed Customer Table & Download
                st.divider()
                st.subheader(f"📋 Customer Details Table ({total_cust} customers)")
                st.dataframe(matched, use_container_width=True)

                csv_bytes = matched.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Customer Segment Predictions as CSV",
                    data=csv_bytes,
                    file_name="matching_customers_churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ==========================================
# TAB 2: Single Customer Predictor
# ==========================================
with tab2:
    st.subheader("👤 Individual Customer Churn Risk Simulator")
    st.markdown("Input specific customer attributes to calculate instant churn risk and generate tailored retention offers.")

    with st.form("single_customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            Partner = st.selectbox("Has Partner", ["No", "Yes"])
            Dependents = st.selectbox("Has Dependents", ["No", "Yes"])

        with col2:
            st.markdown("##### 📄 Contract & Billing")
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=3, step=1)
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=300.0, value=85.50, step=1.0)
            TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=15000.0, value=float(MonthlyCharges * max(1, tenure)), step=10.0)

        with col3:
            st.markdown("##### 🌐 Subscribed Services")
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        single_submit = st.form_submit_button("🔮 Predict Customer Churn Risk", use_container_width=True)

    if single_submit:
        payload = {
            "gender": gender,
            "SeniorCitizen": int(SeniorCitizen),
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": int(tenure),
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": float(MonthlyCharges),
            "TotalCharges": float(TotalCharges)
        }

        proba_val = None
        pred_val = None
        risk_tag = None
        risk_factors_list = []
        source_name = None

        # Try API
        try:
            resp = requests.post(API_URL, json=payload, timeout=2.5)
            if resp.status_code == 200:
                data_json = resp.json()
                pred_val = data_json.get("churn")
                proba_val = data_json.get("probability")
                risk_tag = data_json.get("risk_level")
                risk_factors_list = data_json.get("risk_factors", [])
                source_name = "Flask REST API"
        except Exception:
            pass

        # Fallback to local
        if proba_val is None and pipeline is not None:
            df_in = pd.DataFrame([payload])
            proba_val = float(pipeline.predict_proba(df_in)[:, 1][0])
            pred_val = int(pipeline.predict(df_in)[0])
            risk_tag = "High" if proba_val >= 0.65 else ("Medium" if proba_val >= 0.35 else "Low")
            source_name = "Local ML Pipeline (joblib)"

        if proba_val is not None:
            st.divider()
            st.subheader("🎯 Individual Risk Assessment Results")
            st.caption(f"Engine: {source_name}")

            res1, res2, res3 = st.columns(3)
            with res1:
                if proba_val >= 0.5:
                    st.error("### ⚠️ Churn Risk: HIGH")
                else:
                    st.success("### ✅ Churn Risk: LOW")

            with res2:
                st.metric(
                    label="Churn Probability",
                    value=f"{proba_val * 100:.1f}%",
                    delta=f"{'+' if proba_val >= 0.5 else '-'}{abs(proba_val - 0.5)*100:.1f}% vs baseline",
                    delta_color="inverse"
                )

            with res3:
                st.metric(label="Risk Classification", value=f"{risk_tag} Risk")

            st.progress(proba_val, text=f"Risk Score: {proba_val * 100:.1f} / 100")

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("#### 🔍 Primary Risk Indicators")
                if not risk_factors_list:
                    # generate standard factors
                    if Contract == "Month-to-month":
                        risk_factors_list.append("Month-to-month contract")
                    if tenure < 12:
                        risk_factors_list.append(f"Short tenure ({tenure} months)")
                    if MonthlyCharges > 70:
                        risk_factors_list.append(f"High monthly charges (${MonthlyCharges:.2f})")
                    if TechSupport == "No" and InternetService != "No":
                        risk_factors_list.append("No tech support add-on")
                    if PaymentMethod == "Electronic check":
                        risk_factors_list.append("Electronic check payment")

                if risk_factors_list:
                    for rf in risk_factors_list:
                        st.write(f"- {rf}")
                else:
                    st.write("- Low risk profile attributes.")

            with r_col2:
                st.markdown("#### 💡 Targeted Retention Action Plan")
                recs = get_retention_advice(payload, proba_val)
                for r in recs:
                    st.write(r)


