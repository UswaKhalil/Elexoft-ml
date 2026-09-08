import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ---------- Page setup ----------
st.set_page_config(page_title="Wine Classifier", page_icon="🍷")
st.title("🍷 Wine Cultivar Classifier")
st.write(
    "A small neural network (MLPClassifier) predicts which of 3 grape "
    "cultivars a wine belongs to, based on 13 chemical measurements."
)

# ---------- Load data + train model (cached so it only runs once) ----------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("wine.csv")
    X = df.drop("Wine", axis=1).values
    y = df["Wine"].values - 1  # labels 1,2,3 -> 0,1,2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(16,), activation="relu", max_iter=1000, random_state=42
    )
    model.fit(X_train, y_train)

    test_accuracy = model.score(X_test, y_test)

    return df, scaler, model, test_accuracy


df, scaler, model, test_accuracy = load_and_train()
feature_names = df.drop("Wine", axis=1).columns.tolist()

st.sidebar.header("Wine chemical measurements")
st.sidebar.write(f"Model test accuracy: **{test_accuracy:.2%}**")

# ---------- Sidebar inputs: one slider per feature -----------
user_values = {}
for col in feature_names:
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    col_mean = float(df[col].mean())
    user_values[col] = st.sidebar.slider(
        col, min_value=col_min, max_value=col_max, value=col_mean
    )

# ---------- Predict ----------
if st.sidebar.button("Predict wine class"):
    input_row = np.array([[user_values[col] for col in feature_names]])
    input_scaled = scaler.transform(input_row)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.subheader(f"Predicted cultivar: **Class {prediction + 1}**")

    prob_df = pd.DataFrame(
        {"Class": [f"Class {i + 1}" for i in range(len(probabilities))],
         "Probability": probabilities}
    ).set_index("Class")
    st.bar_chart(prob_df)

else:
    st.info("Adjust the sliders in the sidebar, then click **Predict wine class**.")

# ---------- Show the raw data if the user wants to explore ----------
with st.expander("View the training dataset"):
    st.dataframe(df)
  
