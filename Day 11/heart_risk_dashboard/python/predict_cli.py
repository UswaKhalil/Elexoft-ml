"""
predict_cli.py
---------------
Called by predict.php via shell_exec/proc_open. Reads one JSON object of
patient data from stdin, loads the trained model, and prints the prediction
as JSON to stdout.

Usage (for testing directly):
    echo '{"Chest_Pain":1, ... }' | python predict_cli.py
"""

import sys
import json
import os
import xgboost as xgb
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "heart_risk_best_model.json")

FEATURE_ORDER = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
    "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea",
    "High_BP", "High_Cholesterol", "Diabetes", "Smoking", "Obesity",
    "Sedentary_Lifestyle", "Family_History", "Chronic_Stress", "Gender", "Age",
]


def main():
    raw = sys.stdin.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON input."}))
        return

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        print(json.dumps({"error": f"Missing fields: {missing}"}))
        return

    try:
        row = pd.DataFrame([[float(data[f]) for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    except (ValueError, TypeError):
        print(json.dumps({"error": "All fields must be numeric."}))
        return

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    print(json.dumps({
        "prediction": prediction,
        "label": "High Risk" if prediction == 1 else "Low Risk",
        "probability_high_risk": round(probability, 4),
    }))


if __name__ == "__main__":
    main()
