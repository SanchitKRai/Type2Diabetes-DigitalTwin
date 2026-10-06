import joblib
import pandas as pd
import shap


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_PATH = "data/processed/ml_ready.csv"
MODEL_PATH = "models/xgb_final.pkl"
OUTPUT_PATH = "data/processed/demo_shap_explanations.csv"


# ---------------------------------------------------------
# Model features
# ---------------------------------------------------------
model_features = [
    "glucose_mg_dl",
    "glucose_change_15min",
    "glucose_rolling_mean_30min",
    "glucose_rolling_std_30min",
    "steps",
    "steps_30min",
    "sleep_stage_encoded",
    "gender_encoded",
    "age",
    "hba1c",
    "bmi",
    "baseline_glucose",
    "cholesterol",
    "triglycerides",
    "medication_count",
    "hour",
    "minute",
]


# ---------------------------------------------------------
# Load ML-ready data and model
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

print(f"Loaded dataset: {df.shape}")


# ---------------------------------------------------------
# Demo patients
# ---------------------------------------------------------
demo_patients = [
    "187a1094-82b0-8949-3e8e-1b78c62581c1",
    "d7af6e65-bec0-a9a9-a7d6-fa39bdfa460d",
    "db55a107-0cb4-0276-8ade-7d4a0a0dd424",
    "327c1aa7-a065-a2e1-4bd7-5c2a1b0ee87a",
    "09f51a16-d166-d7b3-1965-c9f60181446a",
]

demo_df = df[df["patient_id"].isin(demo_patients)].copy()

print(f"Demo rows: {len(demo_df)}")
print(f"Demo patients found: {demo_df['patient_id'].nunique()}")


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------
X = demo_df[model_features]

demo_df["predicted_spike_probability"] = (
    model.predict_proba(X)[:, 1]
)

threshold = 0.64

demo_df["predicted_spike"] = (
    demo_df["predicted_spike_probability"] >= threshold
).astype(int)


# ---------------------------------------------------------
# Select highest-risk timestamp for each patient
# ---------------------------------------------------------
demo_rows = (
    demo_df
    .sort_values(
        ["patient_id", "predicted_spike_probability"],
        ascending=[True, False]
    )
    .groupby("patient_id")
    .head(1)
    .copy()
)


print("\nSelected high-risk timestamps:")
print(
    demo_rows[
        [
            "patient_id",
            "timestamp",
            "glucose_mg_dl",
            "predicted_spike_probability",
            "glucose_spike_2h",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# SHAP
# ---------------------------------------------------------
explainer = shap.TreeExplainer(model)

X_demo = demo_rows[model_features]

shap_values = explainer.shap_values(X_demo)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_values = pd.DataFrame(
    shap_values,
    columns=model_features,
    index=demo_rows.index
)


# ---------------------------------------------------------
# Build clean explanation table
# ---------------------------------------------------------
output_rows = []

for idx, row in demo_rows.iterrows():

    patient_shap = shap_values.loc[idx]

    top_features = (
        patient_shap.abs()
        .sort_values(ascending=False)
        .head(3)
        .index
    )

    output = {
        "patient_id": row["patient_id"],
        "timestamp": row["timestamp"],
        "current_glucose": row["glucose_mg_dl"],
        "predicted_probability": row["predicted_spike_probability"],
        "predicted_spike": row["predicted_spike"],
        "actual_spike": row["glucose_spike_2h"],
    }

    for i, feature in enumerate(top_features, start=1):

        output[f"top_feature_{i}"] = feature
        output[f"top_feature_{i}_shap"] = patient_shap[feature]

    output_rows.append(output)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
result = pd.DataFrame(output_rows)

result.to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------
print("\n===== DEMO SHAP EXPLANATIONS =====\n")

print(result.to_string(index=False))

print("\n===================================")
print(f"Patients: {len(result)}")
print(f"Saved: {OUTPUT_PATH}")