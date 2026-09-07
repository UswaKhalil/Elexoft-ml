"""
Customer Churn Prediction & Segmentation App
Run with: streamlit run app.py

Requires these files in the same folder (produced by the capstone notebook):
    churn_model.pkl, kmeans_model.pkl, cluster_scaler.pkl, feature_scaler.pkl,
    feature_columns.pkl, cluster_names.pkl, numeric_cols.pkl, categorical_cols.pkl,
    WA_Fn-UseC_-Telco-Customer-Churn.csv  (used only for the dashboard page)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Prediction & Segmentation", layout="wide")

# ---------------------------------------------------------------------------
# Load saved artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    churn_model = joblib.load("churn_model.pkl")
    kmeans_model = joblib.load("kmeans_model.pkl")
    cluster_scaler = joblib.load("cluster_scaler.pkl")
    feature_scaler = joblib.load("feature_scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    cluster_names = joblib.load("cluster_names.pkl")
    numeric_cols = joblib.load("numeric_cols.pkl")
    categorical_cols = joblib.load("categorical_cols.pkl")
    return (churn_model, kmeans_model, cluster_scaler, feature_scaler,
            feature_columns, cluster_names, numeric_cols, categorical_cols)

@st.cache_data
def load_reference_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan).astype(float).fillna(0)
    service_cols = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                     "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    df["NumServices"] = df[service_cols].apply(lambda row: sum(v == "Yes" for v in row), axis=1)
    return df

(churn_model, kmeans_model, cluster_scaler, feature_scaler,
 feature_columns, cluster_names, numeric_cols, categorical_cols) = load_artifacts()

ref_df = load_reference_data()

CLUSTER_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Navigate", ["Predict Churn", "Dashboard"])

# ---------------------------------------------------------------------------
# PAGE 1: Predict Churn
# ---------------------------------------------------------------------------
if page == "Predict Churn":
    st.title("Customer Churn Prediction")
    st.write("Enter customer details to predict churn risk and see their segment.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)

    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                     value=float(monthly_charges * max(tenure, 1)))

    if st.button("Predict", type="primary"):
        service_flags = [phone_service, multiple_lines, online_security, online_backup,
                          device_protection, tech_support, streaming_tv, streaming_movies]
        num_services = sum(v == "Yes" for v in service_flags)

        raw_input = {
            "gender": gender, "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
            "NumServices": num_services,
        }
        input_df = pd.DataFrame([raw_input])

        # --- Segment prediction ---
        cluster_input = cluster_scaler.transform(input_df[CLUSTER_FEATURES])
        cluster_id = int(kmeans_model.predict(cluster_input)[0])
        segment_name = cluster_names[cluster_id]
        input_df["Cluster"] = cluster_id

        # --- Churn prediction ---
        encoded = pd.get_dummies(input_df, columns=categorical_cols)
        encoded = encoded.reindex(columns=feature_columns, fill_value=0)
        encoded[numeric_cols] = feature_scaler.transform(encoded[numeric_cols])

        pred = churn_model.predict(encoded)[0]
        proba = churn_model.predict_proba(encoded)[0][1]

        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if pred == 1:
                st.error(f"**Prediction: Likely to Churn**\n\nProbability: {proba:.1%}")
            else:
                st.success(f"**Prediction: Likely to Stay**\n\nProbability of churn: {proba:.1%}")
        with res_col2:
            st.info(f"**Customer Segment:** {segment_name}")

# ---------------------------------------------------------------------------
# PAGE 2: Dashboard
# ---------------------------------------------------------------------------
else:
    st.title("Churn Dashboard")

    overall_rate = (ref_df["Churn"] == "Yes").mean()
    st.metric("Overall Churn Rate", f"{overall_rate:.1%}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Contract Type")
        contract_churn = ref_df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean())
        fig, ax = plt.subplots()
        contract_churn.plot(kind="bar", color="steelblue", ax=ax)
        ax.set_ylabel("Churn Rate")
        ax.set_xlabel("")
        plt.xticks(rotation=20)
        st.pyplot(fig)

    with col2:
        st.subheader("Churn Rate by Segment")
        cluster_input_all = cluster_scaler.transform(ref_df[CLUSTER_FEATURES])
        ref_df_local = ref_df.copy()
        ref_df_local["Segment"] = pd.Series(kmeans_model.predict(cluster_input_all)).map(cluster_names)
        segment_churn = ref_df_local.groupby("Segment")["Churn"].apply(lambda s: (s == "Yes").mean())
        fig2, ax2 = plt.subplots()
        segment_churn.sort_values().plot(kind="barh", color="coral", ax=ax2)
        ax2.set_xlabel("Churn Rate")
        st.pyplot(fig2)

    st.subheader("Segment Sizes")
    st.bar_chart(ref_df_local["Segment"].value_counts())
