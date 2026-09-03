# Heart Risk Assessment — XAMPP version

Runs entirely under Apache + PHP (XAMPP). PHP calls a Python script behind
the scenes to get predictions from your trained XGBoost model — no separate
Flask server needed.

## How it works

```
Browser (index.html)
   │  fetch("predict.php")
   ▼
predict.php (PHP)
   │  proc_open() — sends patient data to Python over stdin
   ▼
python/predict_cli.py
   │  loads heart_risk_best_model.joblib, predicts, prints JSON
   ▼
predict.php relays that JSON back to the browser
```

## Folder structure

```
heart_risk/
├── index.html              ← the page (form + result panel)
├── predict.php              ← PHP bridge to Python
├── assets/
│   ├── style.css
│   └── script.js
└── python/
    ├── predict_cli.py       ← loads the model, does the prediction
    ├── requirements.txt
    └── model/
        └── heart_risk_best_model.json   ← portable XGBoost format
```

## Setup steps

1. **Copy the `heart_risk` folder into XAMPP's `htdocs`.**
   - Windows: `C:\xampp\htdocs\heart_risk`
   - Mac: `/Applications/XAMPP/htdocs/heart_risk`

2. **Make sure Python is installed on the same machine** and install the
   model's dependencies:
   ```
   pip install -r python/requirements.txt
   ```

3. **Confirm the Python command works from a terminal.** Open a terminal
   and try:
   ```
   python --version
   ```
   If that doesn't work, try `python3` or `py` instead. Whatever works,
   open `predict.php` and update this line near the top to match:
   ```php
   $pythonExecutable = "python"; // <-- change this if needed
   ```
   On Windows, if `python` isn't recognized at all, use the full path, e.g.:
   ```php
   $pythonExecutable = "C:\\Python312\\python.exe";
   ```

4. **Start Apache** in the XAMPP control panel (you don't need MySQL for
   this).

5. **Open your browser to:**
   ```
   http://localhost/heart_risk/
   ```

## Testing the Python side on its own

Before touching PHP, you can sanity-check the model script directly from a
terminal — this isolates whether a problem is in Python or in PHP:

```
cd python
echo {"Chest_Pain":1,"Shortness_of_Breath":1,"Fatigue":1,"Palpitations":0,"Dizziness":0,"Swelling":0,"Pain_Arms_Jaw_Back":1,"Cold_Sweats_Nausea":1,"High_BP":1,"High_Cholesterol":1,"Diabetes":1,"Smoking":1,"Obesity":1,"Sedentary_Lifestyle":1,"Family_History":1,"Chronic_Stress":1,"Gender":1,"Age":65} | python predict_cli.py
```
You should get back something like:
```json
{"prediction": 1, "label": "High Risk", "probability_high_risk": 1.0}
```

## Troubleshooting

- **Blank/500 error from predict.php** — check `python_stderr` in the JSON
  response it returns; it'll usually say exactly what's missing (e.g. a
  missing pip package).
- **"Failed to start the Python process"** — `$pythonExecutable` in
  `predict.php` isn't a valid command on your system. Fix it per step 3.
- **Works in Python but not through the browser** — some Windows setups run
  Apache as a service with a different PATH than your terminal. Using the
  full path to `python.exe` in `$pythonExecutable` avoids this entirely.
- **`XGBoostError: input stream corrupted`** — this means the XGBoost
  version that loads the model doesn't match the version that saved it.
  The model is stored in `heart_risk_best_model.json` (XGBoost's own
  portable format, not pickle) specifically to avoid this, so make sure
  `requirements.txt` was installed with the pinned version:
  `pip install -r python/requirements.txt`

## Notes

- `Gender` is encoded 0/1 in training data; the form assumes 0 = Female,
  1 = Male — flip it in `index.html` if your source used the opposite.
- This is a demo/learning tool trained on a dataset with very clean,
  high-accuracy labels (likely rule-generated, not real clinical outcomes)
  — not intended for actual medical decisions.
