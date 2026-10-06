import pandas as pd
import joblib
from pathlib import Path

try:
    import shap
except ImportError:
    shap = None
    import xgboost as xgb

# -----------------------------------------
# Project paths
# -----------------------------------------

project_root = Path(__file__).resolve().parents[1]

data_file = (
    project_root
    / "data"
    / "processed"
    / "ml_ready.csv"
)

model_file = (
    project_root
    / "models"
    / "xgb_final.pkl"
)

demo_file = (
    project_root
    / "data"
    / "processed"
    / "demo_predictions.csv"
)

# -----------------------------------------
# Load data and final model
# -----------------------------------------

df = pd.read_csv(data_file)

model = joblib.load(model_file)

demo_df = pd.read_csv(demo_file)

print("Data loaded.")
print("Dataset shape:", df.shape)
print("Demo rows:", len(demo_df))

# -----------------------------------------
# Model features
# -----------------------------------------

features = [
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
    "minute"
]

# -----------------------------------------
# Create SHAP explainer
# -----------------------------------------

if shap is not None:
    explainer = shap.TreeExplainer(model)
    print("\nSHAP explainer created.")
else:
    print("\nSHAP package unavailable; using XGBoost native TreeSHAP contributions.")

# -----------------------------------------
# Global SHAP analysis
# -----------------------------------------

X = df[features]

if shap is not None:
    shap_values = explainer.shap_values(X)
else:
    shap_values = model.get_booster().predict(
        xgb.DMatrix(X),
        pred_contribs=True
    )[:, :-1]

global_importance = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
})

global_importance = global_importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

print("\n===== GLOBAL SHAP IMPORTANCE =====")
print(global_importance.to_string(index=False))

# -----------------------------------------
# Local SHAP explanations
# -----------------------------------------

print("\n===== LOCAL SHAP EXPLANATIONS =====")

# Select one row per demo patient
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

# Match rows against original ML dataset
demo_features = df[
    df["patient_id"].isin(demo_rows["patient_id"])
].copy()

# Use the same patient/timestamp combination
demo_features["timestamp"] = pd.to_datetime(
    demo_features["timestamp"]
)

demo_rows["timestamp"] = pd.to_datetime(
    demo_rows["timestamp"]
)

demo_features = demo_features.merge(
    demo_rows[
        [
            "patient_id",
            "timestamp",
            "predicted_spike_probability",
            "predicted_spike",
            "actual_spike_2h"
        ]
    ],
    on=["patient_id", "timestamp"],
    how="inner"
)

X_demo = demo_features[features]

if shap is not None:
    local_shap_values = explainer.shap_values(X_demo)
else:
    local_shap_values = model.get_booster().predict(
        xgb.DMatrix(X_demo),
        pred_contribs=True
    )[:, :-1]

for i in range(len(demo_features)):

    row = demo_features.iloc[i]

    shap_row = local_shap_values[i]

    explanation = pd.DataFrame({
        "feature": features,
        "shap_value": shap_row,
        "abs_shap": abs(shap_row)
    })

    explanation = explanation.sort_values(
        "abs_shap",
        ascending=False
    )

    print("\n-----------------------------------------")
    print("Patient:", row["patient_id"])
    print(
        f"Prediction probability: "
        f"{row['predicted_spike_probability']:.3f}"
    )
    print(
        "Predicted spike:",
        int(row["predicted_spike"])
    )
    print(
        "Actual spike:",
        int(row["actual_spike_2h"])
    )

    print("\nTop 3 SHAP contributors:")

    print(
        explanation[
            ["feature", "shap_value"]
        ].head(3).to_string(index=False)
    )