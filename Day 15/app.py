import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Wine Group Predictor", layout="centered")
st.title("Wine Group Predictor")
st.caption("Enter a wine's chemical measurements below to see which group it's predicted to belong to.")


# ---------------------------------------------------------------------------
# Train the model once (cached) on the built-in wine.csv
# ---------------------------------------------------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("wine.csv")
    X = df.drop(columns=["Wine"])
    y = df["Wine"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    grid = GridSearchCV(
        estimator=Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
        param_grid={
            "knn__n_neighbors": list(range(1, 21)),
            "knn__weights": ["uniform", "distance"],
        },
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, X, df


model, X, df = train_model()
feature_names = list(X.columns)
groups = sorted(df["Wine"].unique())

# ---------------------------------------------------------------------------
# Show the 3 wine classes at the top
# ---------------------------------------------------------------------------
st.subheader("The 3 wine classes")
st.caption(
    "This model predicts which of these three wine classes (cultivars) a sample belongs to, "
    "based on its chemical measurements. Each class has its own typical chemical profile:"
)

profile_features = ["Flavanoids", "Color.int", "Proline"]
class_profiles = df.groupby("Wine")[profile_features].mean()

group_cols = st.columns(len(groups))
for gcol, group_id in zip(group_cols, groups):
    count = (df["Wine"] == group_id).sum()
    with gcol:
        st.markdown(f"**Class {group_id}**")
        st.caption(f"{count} training samples")
        for feature in profile_features:
            st.metric(feature, f"{class_profiles.loc[group_id, feature]:.2f}")

st.divider()

# ---------------------------------------------------------------------------
# User input for each feature
# ---------------------------------------------------------------------------
st.subheader("Enter the wine's measurements")

user_values = {}
cols = st.columns(2)
for i, feature in enumerate(feature_names):
    col = cols[i % 2]
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    with col:
        user_values[feature] = st.number_input(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
        )

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Predict group", type="primary"):
    input_df = pd.DataFrame([user_values])[feature_names]
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted wine class: **Class {prediction}**")

    if hasattr(model.named_steps["knn"], "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        proba_df = pd.DataFrame({"Class": groups, "Probability": proba}).set_index("Class")
        st.bar_chart(proba_df)
