"""
train_save_model.py
Trains the ANN model using the EXACT same logic as your original code:
  - Same Risk Level creation (injury_score + damage_score)
  - Same 13 features
  - Same preprocessing
  - Same 3 classes: Low / Medium / High
Run this ONCE before starting the Flask app.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import classification_report, confusion_matrix, accuracy_score

np.random.seed(42)

# ── STEP 1: Load Dataset ──────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("traffic_accidents.csv")
print(f"Rows: {len(df)}  |  Columns: {len(df.columns)}")

# ── STEP 2: Create Risk Level Target (same as your code) ─────
injury_score_map = {
    "NO INDICATION OF INJURY"  : 0,
    "REPORTED, NOT EVIDENT"    : 1,
    "NONINCAPACITATING INJURY" : 2,
    "INCAPACITATING INJURY"    : 3,
    "FATAL"                    : 4
}
damage_score_map = {
    "$500 OR LESS"  : 0,
    "$501 - $1,500" : 1,
    "OVER $1,500"   : 2
}

df["injury_score"] = df["most_severe_injury"].map(injury_score_map).fillna(0)
df["damage_score"] = df["damage"].map(damage_score_map).fillna(0)
df["total_score"]  = df["injury_score"] + df["damage_score"]

def get_risk_label(score):
    if score <= 1:   return "Low"
    elif score <= 3: return "Medium"
    else:            return "High"

df["Risk_Level"] = df["total_score"].apply(get_risk_label)

print("\nRisk Level distribution:")
print(df["Risk_Level"].value_counts())

# ── STEP 3: Select Features (same 13 as your code) ───────────
feature_columns = [
    "weather_condition",
    "lighting_condition",
    "roadway_surface_cond",
    "road_defect",
    "trafficway_type",
    "alignment",
    "traffic_control_device",
    "first_crash_type",
    "intersection_related_i",
    "num_units",
    "crash_hour",
    "crash_day_of_week",
    "crash_month"
]

X = df[feature_columns].copy()
y = df["Risk_Level"].copy()

# ── STEP 4: Encode Categorical Columns ───────────────────────
encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print("\nTarget classes:", list(target_encoder.classes_))

# ── STEP 5: Normalize ─────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── STEP 6: Train / Test Split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ── STEP 7 & 8: Build & Train ANN ────────────────────────────
# Same architecture as your code: 64 → 32 → 3 (Low/Med/High)
print("\nTraining ANN model...")
model = MLPClassifier(
    hidden_layer_sizes  = (64, 32),      # same as your Dense(64) → Dense(32)
    activation          = 'relu',
    solver              = 'adam',
    alpha               = 0.001,         # L2 regularization (like Dropout)
    batch_size          = 32,
    learning_rate_init  = 0.001,
    max_iter            = 100,
    early_stopping      = True,
    validation_fraction = 0.2,
    n_iter_no_change    = 10,
    random_state        = 42,
    verbose             = False
)
model.fit(X_train, y_train)

# ── STEP 9: Evaluate ─────────────────────────────────────────
y_pred    = model.predict(X_test)
train_acc = round(accuracy_score(y_train, model.predict(X_train)) * 100, 2)
test_acc  = round(accuracy_score(y_test,  y_pred) * 100, 2)

print(f"\nTrain Accuracy : {train_acc}%")
print(f"Test  Accuracy : {test_acc}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=target_encoder.classes_))

# ── Save unique values for dropdowns ─────────────────────────
cat_cols    = list(encoders.keys())
unique_vals = {}
for col in cat_cols:
    unique_vals[col] = [str(v) for v in encoders[col].classes_]

# ── Save loss curve ───────────────────────────────────────────
loss_curve = [round(v, 4) for v in model.loss_curve_]
val_scores = [round(v, 4) for v in (model.validation_scores_ or [])]

# ── Confusion matrix ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred).tolist()

# ── Risk distribution ─────────────────────────────────────────
risk_counts = df["Risk_Level"].value_counts()
risk_dist   = {
    "Low"   : int(risk_counts.get("Low",    0)),
    "Medium": int(risk_counts.get("Medium", 0)),
    "High"  : int(risk_counts.get("High",   0)),
}

# ── Save all metadata ─────────────────────────────────────────
meta = {
    "features"      : feature_columns,
    "cat_cols"      : cat_cols,
    "classes"       : list(target_encoder.classes_),
    "unique_vals"   : unique_vals,
    "train_acc"     : train_acc,
    "test_acc"      : test_acc,
    "epochs"        : model.n_iter_,
    "loss_curve"    : loss_curve,
    "val_scores"    : val_scores,
    "cm"            : cm,
    "risk_dist"     : risk_dist,
}

joblib.dump(model,          "model.pkl")
joblib.dump(scaler,         "scaler.pkl")
joblib.dump(encoders,       "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
with open("meta.json", "w") as f:
    json.dump(meta, f)

print("\n" + "="*45)
print("  Model saved successfully!")
print(f"  Train Accuracy : {train_acc}%")
print(f"  Test  Accuracy : {test_acc}%")
print(f"  Classes        : {list(target_encoder.classes_)}")
print("  Files: model.pkl, scaler.pkl, encoders.pkl,")
print("         target_encoder.pkl, meta.json")
print("="*45)
print("Now run: python app.py")
