# 🌙 Sleep Disorder Prediction — Capstone Project

Predicting whether a person shows patterns associated with **Insomnia**,
**Sleep Apnea**, or **no sleep disorder**, based on lifestyle and health data
(sleep duration, stress level, physical activity, blood pressure, occupation,
and more).

This project covers the full pipeline: data cleaning, feature engineering,
exploratory data analysis, model training/comparison, hyperparameter tuning,
and a deployed interactive web app.

---

## 📁 Project Structure

```
├── Capstone_Project.ipynb      # Full analysis notebook (EDA, cleaning, modeling)
├── Sleep_and_Lifestyle_Dataset.csv   # Source dataset
├── app.py                      # Streamlit web app
├── sleep_disorder_model.pkl    # Trained & tuned Random Forest model
├── scaler.pkl                  # Fitted StandardScaler
├── feature_columns.pkl         # Exact feature column order used in training
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧠 What's Inside the Notebook

1. **Data loading & exploration** — shape, dtypes, summary statistics.
2. **Cleaning** — handling missing `Sleep Disorder` values, removing
   duplicates, splitting `Blood Pressure` into `U_BP` (systolic) and `L_BP`
   (diastolic).
3. **Encoding** — mapping `Gender` to 0/1, one-hot encoding `Occupation` and
   `BMI Category`.
4. **Visualization** — distribution of sleep duration, sleep disorder
   counts, correlation heatmap.
5. **Modeling** — training and comparing Logistic Regression, KNN, and
   Random Forest.
6. **Tuning** — `GridSearchCV` over Random Forest hyperparameters
   (`n_estimators`, `max_depth`).
7. **Saving** — exporting the final tuned model, scaler, and feature
   columns with `joblib` for use in the app.

**Best model:** Random Forest (tuned), ~92.5% accuracy on the held-out test set.

---

## 🚀 Running the App Locally

1. **Clone/download this project folder** so all files sit in the same
   directory (`app.py` needs `sleep_disorder_model.pkl`, `scaler.pkl`, and
   `feature_columns.pkl` right next to it).

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. Your browser will open automatically at `http://localhost:8501`.

---

## 🖥️ How the App Works

- The sidebar collects the same inputs used during training: demographics
  (gender, age, occupation, BMI category), sleep & activity habits (sleep
  duration, quality, physical activity, stress, daily steps), and vitals
  (heart rate, systolic/diastolic blood pressure).
- These inputs are transformed into the **exact same feature format** the
  model was trained on (same one-hot encoded columns, same scaling), using
  the saved `feature_columns.pkl` and `scaler.pkl`.
- The model returns a prediction plus a probability breakdown across all
  three classes, shown as a bar chart alongside the final result.

---

## ⚠️ Disclaimer

This app is a machine learning demo built for a capstone/educational
project. It is **not a medical diagnostic tool**. Predictions should not be
used to make real health decisions — please consult a healthcare
professional for any genuine sleep or health concerns.

---

## 🛠️ Tech Stack

- **Python**, **pandas**, **scikit-learn** — data processing & modeling
- **Streamlit** — interactive web app
- **joblib** — model persistence
