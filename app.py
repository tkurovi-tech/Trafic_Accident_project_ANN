import os
"""
app.py  —  Traffic Accident Risk Prediction Flask App
4 Pages: Home / Predict / Dashboard / Model Info
Run: python app.py  then open  http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# ── Load saved model artifacts ────────────────────────────────
BASE           = os.path.dirname(__file__)
model          = joblib.load(os.path.join(BASE, "model.pkl"))
scaler         = joblib.load(os.path.join(BASE, "scaler.pkl"))
encoders       = joblib.load(os.path.join(BASE, "encoders.pkl"))
target_encoder = joblib.load(os.path.join(BASE, "target_encoder.pkl"))
with open(os.path.join(BASE, "meta.json")) as f:
    meta = json.load(f)

FEATURES    = meta["features"]
UNIQUE_VALS = meta["unique_vals"]
CLASSES     = meta["classes"]   # ['High', 'Low', 'Medium'] (alphabetical)

# ── PAGES ─────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html", meta=meta)

@app.route("/predict")
def predict_page():
    return render_template("predict.html", unique_vals=UNIQUE_VALS)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", meta=meta)

@app.route("/model-info")
def model_info():
    return render_template("model_info.html", meta=meta)

# ── PREDICTION API ────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    row  = []

    for feat in FEATURES:
        val = data.get(feat, 0)
        if feat in encoders:
            val = str(val).upper()
            cls = encoders[feat].classes_
            val = encoders[feat].transform([val])[0] if val in cls \
                  else encoders[feat].transform([cls[0]])[0]
        else:
            try:    val = float(val)
            except: val = 0
        row.append(val)

    X_new  = scaler.transform([row])
    probs  = model.predict_proba(X_new)[0]
    pred   = int(model.predict(X_new)[0])
    label  = target_encoder.inverse_transform([pred])[0]
    conf   = round(float(np.max(probs)) * 100, 1)

    # Build probability dict for all 3 classes
    prob_dict = {}
    for cls, p in zip(CLASSES, probs):
        prob_dict[cls] = round(float(p) * 100, 1)

    return jsonify({
        "prediction"  : pred,
        "label"       : label,
        "confidence"  : conf,
        "probabilities": prob_dict,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
