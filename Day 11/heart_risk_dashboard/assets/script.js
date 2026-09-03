const form = document.getElementById("intake-form");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const resultError = document.getElementById("result-error");
const resultErrorText = document.getElementById("result-error-text");

const heartEl = document.getElementById("result-heart");
const verdictEl = document.getElementById("result-verdict");
const probabilityEl = document.getElementById("result-probability");
const gaugeFillEl = document.getElementById("gauge-fill");
const explainerEl = document.getElementById("result-explainer");

const CHECKBOX_FIELDS = [
  "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
  "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea",
  "High_BP", "High_Cholesterol", "Diabetes", "Smoking", "Obesity",
  "Sedentary_Lifestyle", "Family_History", "Chronic_Stress",
];

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {};
  CHECKBOX_FIELDS.forEach((name) => {
    payload[name] = form.elements[name].checked ? 1 : 0;
  });
  payload["Age"] = Number(form.elements["Age"].value);
  payload["Gender"] = Number(form.querySelector('input[name="Gender"]:checked').value);

  setLoading(true);

  try {
    const response = await fetch("predict.php", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || "Something went wrong while running the assessment.");
      return;
    }

    showResult(data);
  } catch (err) {
    showError("Couldn't reach predict.php. Is Apache running in XAMPP?");
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  const btn = form.querySelector(".run-btn");
  btn.disabled = isLoading;
  btn.textContent = isLoading ? "Running assessment…" : "Run assessment";
}

function showResult(data) {
  resultEmpty.hidden = true;
  resultError.hidden = true;
  resultContent.hidden = false;

  const isHigh = data.prediction === 1;
  const pct = Math.round(data.probability_high_risk * 100);

  // retrigger the heartbeat pulse animation
  heartEl.classList.remove("beat");
  void heartEl.offsetWidth; // force reflow so the animation restarts
  heartEl.classList.add("beat");

  verdictEl.textContent = data.label;
  verdictEl.className = "result__verdict " + (isHigh ? "high" : "low");

  probabilityEl.textContent = `${pct}%`;

  gaugeFillEl.className = "gauge__fill " + (isHigh ? "high" : "low");
  // reset then animate on next frame so the transition plays
  gaugeFillEl.style.width = "0%";
  requestAnimationFrame(() => {
    gaugeFillEl.style.width = `${pct}%`;
  });

  explainerEl.textContent = isHigh
    ? "The model flags this combination of symptoms and risk factors as high risk. This is a model prediction, not a diagnosis — recommend clinical follow-up."
    : "The model flags this combination of symptoms and risk factors as low risk. This is a model prediction, not a diagnosis.";
}

function showError(message) {
  resultEmpty.hidden = true;
  resultContent.hidden = true;
  resultError.hidden = false;
  resultErrorText.textContent = message;
}
