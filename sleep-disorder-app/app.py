import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="🌙",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Light custom styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(180deg, #0f1225 0%, #171a34 100%);
    }
    h1, h2, h3 {
        color: #e8e6ff;
    }
    p, label, .stMarkdown {
        color: #cfd2e8;
    }
    div[data-testid="stMetricValue"] {
        color: #a78bfa;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        background: rgba(167, 139, 250, 0.08);
        border: 1px solid rgba(167, 139, 250, 0.35);
        text-align: center;
        margin-top: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .badge-none { background: #16532b; color: #b7f7c8; }
    .badge-apnea { background: #5c2d0c; color: #ffd6a5; }
    .badge-insomnia { background: #4a1d3d; color: #ffb3e6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model artifacts
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("sleep_disorder_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

OCCUPATIONS = [
    "Accountant", "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
    "Sales Representative", "Salesperson", "Scientist", "Software Engineer", "Teacher",
]
BMI_CATEGORIES = ["Normal", "Normal Weight", "Overweight", "Obese"]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🌙 Sleep Disorder Predictor")
st.write(
    "Answer a few questions about your lifestyle and health, and this app will "
    "estimate whether you show patterns associated with **Insomnia**, "
    "**Sleep Apnea**, or **no sleep disorder**, based on a Random Forest model "
    "trained on sleep & lifestyle data."
)
st.divider()

# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
st.sidebar.header("Your Details")

gender = st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True)
age = st.sidebar.slider("Age", 18, 70, 35)
occupation = st.sidebar.selectbox("Occupation", OCCUPATIONS)
bmi_category = st.sidebar.selectbox("BMI Category", BMI_CATEGORIES)

st.sidebar.header("Sleep & Activity")
sleep_duration = st.sidebar.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, 0.1)
quality_of_sleep = st.sidebar.slider("Quality of Sleep (1-10)", 1, 10, 7)
physical_activity = st.sidebar.slider("Physical Activity Level (0-100)", 0, 100, 50)
stress_level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
daily_steps = st.sidebar.slider("Daily Steps", 1000, 12000, 6500, 100)

st.sidebar.header("Vitals")
heart_rate = st.sidebar.slider("Heart Rate (bpm)", 55, 100, 70)
systolic_bp = st.sidebar.slider("Systolic Blood Pressure", 100, 160, 120)
diastolic_bp = st.sidebar.slider("Diastolic Blood Pressure", 60, 100, 80)

predict_clicked = st.sidebar.button("🔍 Predict", use_container_width=True)

# ----------------------------------------------------------------------------
# Build the model-ready input row
# ----------------------------------------------------------------------------
def build_input_row():
    row = pd.DataFrame(0, index=[0], columns=feature_columns)

    row["Gender"] = 0 if gender == "Male" else 1
    row["Age"] = age
    row["Sleep Duration"] = sleep_duration
    row["Quality of Sleep"] = quality_of_sleep
    row["Physical Activity Level"] = physical_activity
    row["Stress Level"] = stress_level
    row["Heart Rate"] = heart_rate
    row["Daily Steps"] = daily_steps
    row["U_BP"] = systolic_bp
    row["L_BP"] = diastolic_bp

    occ_col = f"Occupation_{occupation}"
    if occ_col in row.columns:
        row[occ_col] = 1

    bmi_col = f"BMI Category_{bmi_category}"
    if bmi_col in row.columns:
        row[bmi_col] = 1

    return row

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
LABEL_STYLE = {
    "None": ("badge-none", "No Sleep Disorder", "✅"),
    "Sleep Apnea": ("badge-apnea", "Sleep Apnea", "⚠️"),
    "Insomnia": ("badge-insomnia", "Insomnia", "🌀"),
}

if predict_clicked:
    input_row = build_input_row()
    input_scaled = scaler.transform(input_row)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    classes = model.classes_

    css_class, display_name, emoji = LABEL_STYLE.get(
        prediction, ("badge-none", prediction, "🔎")
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div style="font-size: 2rem;">{emoji}</div>
            <div class="badge {css_class}">{display_name}</div>
            <p style="margin-top: 0.75rem;">Predicted outcome based on the details you provided.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Prediction Confidence")
    prob_df = pd.DataFrame({"Class": classes, "Probability": probabilities}).sort_values(
        "Probability", ascending=False
    )
    st.bar_chart(prob_df.set_index("Class"))

    with st.expander("See the exact values used for this prediction"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": [
                        "Gender", "Age", "Occupation", "BMI Category",
                        "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
                        "Stress Level", "Daily Steps", "Heart Rate",
                        "Systolic BP", "Diastolic BP",
                    ],
                    "Value": [
                        gender, age, occupation, bmi_category,
                        sleep_duration, quality_of_sleep, physical_activity,
                        stress_level, daily_steps, heart_rate,
                        systolic_bp, diastolic_bp,
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "⚠️ This is an educational demo built on a modeling exercise, not a medical "
        "diagnostic tool. Please consult a healthcare professional for any real "
        "sleep or health concerns."
    )
else:
    st.info("👈 Fill in your details in the sidebar and click **Predict** to get a result.")
