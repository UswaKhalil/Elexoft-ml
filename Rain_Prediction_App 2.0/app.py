"""Streamlit app for the rain-prediction capstone — prediction + documentation tabs.

Run with:  streamlit run app.py

Expects these files in the same folder:
  rain_model.joblib, rain_model_meta.joblib,
  chart_1_humidity_hist.png, chart_2_month_rate.png,
  chart_3_correlation.png, chart_4_features.png
"""

import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Will it rain tomorrow?", page_icon="🌧️", layout="wide")

# --- Load model ---------------------------------------------------------------


@st.cache_resource
def load_model():
    return joblib.load("rain_model.joblib"), joblib.load("rain_model_meta.joblib")


model, meta = load_model()

# --- Sidebar navigation --------------------------------------------------------

st.sidebar.title("🌧️ Rain Capstone")
page = st.sidebar.radio("Go to", ["Predict", "How I Built This"])
st.sidebar.caption(f"Model in use: **{meta['best_model']}**")

# ================================================================================
# PAGE 1: PREDICT
# ================================================================================

if page == "Predict":
    st.title("Will it rain tomorrow?")
    st.write(
        "Enter today's weather at an Australian station and this app predicts whether "
        "at least 1mm of rain falls tomorrow."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Place and time")
        location = st.selectbox("Location", meta["locations"])
        month = st.slider("Month", 1, 12, 6)

        st.subheader("Temperature")
        min_temp = st.number_input("Minimum temperature (°C)", -10.0, 40.0, 12.0, 0.1)
        max_temp = st.number_input("Maximum temperature (°C)", -5.0, 50.0, 23.0, 0.1)
        temp_change = st.number_input(
            "Temperature change, 9am to 3pm (°C)", -10.0, 30.0, 4.5, 0.1
        )

        st.subheader("Water")
        rainfall = st.number_input("Rainfall today (mm)", 0.0, 400.0, 0.0, 0.1)
        rain_today = st.radio("Did it rain today (1mm or more)?", ["No", "Yes"], horizontal=True)
        evaporation = st.number_input("Evaporation (mm)", 0.0, 150.0, 4.8, 0.1)

    with col2:
        st.subheader("Sky and pressure")
        sunshine = st.slider("Sunshine (hours)", 0.0, 14.5, 8.0, 0.1)
        cloud_mean = st.slider(
            "Mean cloud cover (oktas, 0 clear to 8 overcast)", 0.0, 8.0, 4.5, 0.5
        )
        humidity_drop = st.slider("Humidity drop, 9am to 3pm (percentage points)", 0, 95, 17)
        pressure_mean = st.number_input("Mean pressure (hPa)", 970.0, 1045.0, 1016.4, 0.1)
        pressure_trend = st.number_input("Pressure trend, 9am to 3pm (hPa)", -20.0, 20.0, -2.4, 0.1)

        st.subheader("Wind")
        wind_gust_dir = st.selectbox("Strongest gust direction", meta["wind_dirs"], index=0)
        wind_gust_speed = st.slider("Strongest gust speed (km/h)", 5, 140, 40)
        wind_dir_9am = st.selectbox("Wind direction 9am", meta["wind_dirs"], index=0)
        wind_dir_3pm = st.selectbox("Wind direction 3pm", meta["wind_dirs"], index=0)
        wind_speed_mean = st.slider("Mean wind speed (km/h)", 0, 90, 16)

    if st.button("Predict", type="primary"):
        row = {
            "Location": location,
            "MinTemp": min_temp,
            "MaxTemp": max_temp,
            "Rainfall": rainfall,
            "Evaporation": evaporation,
            "Sunshine": sunshine,
            "WindGustDir": wind_gust_dir,
            "WindGustSpeed": wind_gust_speed,
            "WindDir9am": wind_dir_9am,
            "WindDir3pm": wind_dir_3pm,
            "RainToday": 1 if rain_today == "Yes" else 0,
            "Month": month,
            "Pressure_Trend": pressure_trend,
            "Temp_Change": temp_change,
            "Humidity_Drop": humidity_drop,
            "Pressure_Mean": pressure_mean,
            "Cloud_Mean": cloud_mean,
            "WindSpeed_Mean": wind_speed_mean,
        }
        # Rebuild the engineered feature exactly as the notebook does.
        row["Damp_Index"] = cloud_mean * (100 - humidity_drop) / 100

        input_data = pd.DataFrame([row])[meta["columns"]]

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"Rain expected tomorrow — {probability:.0%} chance.")
        else:
            st.info(f"No rain expected tomorrow — {probability:.0%} chance of rain.")

        st.progress(float(probability))
        st.caption(
            "The model predicts 'rain' above a 50% probability. It catches roughly half of "
            "genuinely rainy days, so treat a low number as 'probably dry', not 'certainly dry'."
        )

# ================================================================================
# PAGE 2: HOW I BUILT THIS
# ================================================================================

else:
    st.title("How I Built This")
    st.write(
        "A walkthrough of the dataset, the cleaning, the models compared, and why the "
        "final model was chosen — the same reasoning behind the Predict tab."
    )

    st.header("1. The dataset")
    st.markdown(
        """
- **weatherAUS_reduced.csv** — daily weather observations from **49 Australian stations**
- **140,787 rows, 19 columns**; 30 exact duplicates removed before modeling
- **Target:** `RainTomorrow` — 1 if ≥1mm of rain falls the next day, else 0
- **Class balance:** only **22.2%** of days are followed by rain, so a model that always
  guessed "no rain" would already score 77.8% accuracy — that's the number every model
  below has to beat
        """
    )

    st.header("2. Cleaning and feature engineering")
    st.markdown(
        """
- No missing values in this version of the dataset — nothing to fill or drop
- Four text columns (`Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`) all have more
  than two categories, so each was **one-hot encoded** rather than simple-mapped
- Scaling and encoding live inside a scikit-learn `Pipeline`, fit on the training split
  only, so the test set never leaks into the transformation
- **Engineered feature — `Damp_Index`:**
  `Cloud_Mean × (100 − Humidity_Drop) / 100` — a small afternoon humidity drop means
  something different on an overcast day than a clear one; multiplying the two captures
  "cloudy *and* staying damp"
        """
    )

    st.header("3. Exploratory charts")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Humidity Drop", "Rain by Month", "Correlation Heatmap", "Feature Weights"]
    )

    def show_chart(path, caption):
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.warning(f"Chart file not found: {path}")

    with tab1:
        show_chart(
            "chart_1_humidity_hist.png",
            "Humidity_Drop is right-skewed — many days barely dry out between 9am and 3pm, "
            "and those are the days worth watching.",
        )
    with tab2:
        show_chart(
            "chart_2_month_rate.png",
            "Rain is markedly more likely in the Australian winter (Jun-Aug, ~27-29%) than "
            "in the dry spring months.",
        )
    with tab3:
        show_chart(
            "chart_3_correlation.png",
            "Sunshine (-0.33), Cloud_Mean (+0.31) and RainToday (+0.31) are the strongest "
            "correlates of RainTomorrow.",
        )
    with tab4:
        show_chart(
            "chart_4_features.png",
            "Top coefficients of the chosen model — Location and Damp_Index dominate.",
        )

    st.header("4. Models compared")

    results = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "Baseline"],
            "Accuracy": [0.8404, 0.8284, 0.8469, 0.7783],
            "Precision": [0.7104, 0.6900, 0.7652, None],
            "Recall": [0.4729, 0.4098, 0.4460, 0.0000],
            "F1": [0.5678, 0.5142, 0.5635, None],
            "ROC AUC": [0.8536, 0.8274, 0.8687, 0.5000],
        }
    ).set_index("Model")
    st.dataframe(results, use_container_width=True)

    st.subheader("Mean squared error")
    st.markdown("Reported two ways, since MSE isn't a standard classification metric:")
    mse_table = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "Baseline"],
            "MSE (hard predictions)": [0.1596, 0.1716, 0.1531, 0.2217],
            "MSE (probabilities / Brier)": [0.1152, 0.1240, 0.1115, 0.1725],
        }
    ).set_index("Model")
    st.dataframe(mse_table, use_container_width=True)
    st.caption(
        "MSE on hard 0/1 predictions is just the error rate. The Brier score (MSE on "
        "probabilities) is more informative — it rewards confidence when right and "
        "penalizes confident mistakes."
    )

    st.header("5. Why Logistic Regression")
    st.markdown(
        """
Logistic Regression and the Random Forest are separated by less than half a point on F1
(0.568 vs 0.564). They split the remaining metrics: the Random Forest leads on accuracy,
precision, ROC AUC, and Brier score, while Logistic Regression leads on **recall**
(0.473 vs 0.446 — 168 more rainy days caught out of 6,240 test days).

**Chosen model: Logistic Regression**, for two reasons:
1. **Recall matters most here** — missing a rainy day costs more than an unneeded umbrella,
   and both models already miss over half of them, so the one that misses fewer wins.
2. **Explainability as a tiebreaker** — with performance this close, a model whose every
   coefficient can be read and explained beats a 150-tree ensemble, and it trains in under
   a second instead of two minutes.

**Honest caveat:** the Random Forest's higher ROC AUC means it ranks days better overall, so
at a tuned decision threshold (rather than the default 0.5) it would likely overtake on
recall too — that's the first item on the improvement list.
        """
    )

    st.header("6. What I'd improve")
    st.markdown(
        """
- **Tune the decision threshold** instead of accepting 0.5 — both models miss more than
  half of all rainy days, which is the real weakness here
- **Respect time ordering** — the train/test split is random, but these are daily
  observations, so the model currently learns from days that come after some of its test
  days. A date-based split would be a more honest evaluation.
        """
    )
