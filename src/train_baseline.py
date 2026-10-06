import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold

# Project paths
project_root = Path(__file__).resolve().parents[1]
data_file = project_root / "data" / "processed" / "ml_ready.csv"

# Load dataset
df = pd.read_csv(data_file)

print("Dataset loaded!")
print("Shape:", df.shape)
print("Patients:", df["patient_id"].nunique())

# Features used by the model
all_features = [
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

features = all_features.copy()

target = "glucose_spike_2h"

# Get unique patient IDs
patient_ids = df["patient_id"].unique()

# -------------------------------------------------
# Patient-level split
# 80% development patients / 20% final test patients
# -------------------------------------------------

dev_patients, test_patients = train_test_split(
    patient_ids,
    test_size=0.20,
    random_state=42
)

# Split development patients again:
# 80% training / 20% validation
train_patients, val_patients = train_test_split(
    dev_patients,
    test_size=0.20,
    random_state=42
)

# Create datasets
train_df = df[df["patient_id"].isin(train_patients)].copy()
val_df = df[df["patient_id"].isin(val_patients)].copy()
test_df = df[df["patient_id"].isin(test_patients)].copy()

print("\nPatient split:")
print("Training patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Testing patients:", len(test_patients))

print("\nRow split:")
print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Testing rows:", len(test_df))

# Prepare X and y
X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nValidation target distribution:")
print(y_val.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Calculate class imbalance from training data only
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass imbalance:")
print("Negative cases:", negative_count)
print("Positive cases:", positive_count)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# -----------------------------------------
# Hyperparameter candidates
# -----------------------------------------

hyperparameter_configs = [
    {
        "name": "H1",
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.10
    },
    {
        "name": "H2",
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05
    },
    {
        "name": "H3",
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05
    },
    {
        "name": "H4",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05
    },
    {
        "name": "H5",
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03
    }
]

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib

# -----------------------------------------
# 5-Fold Patient-Level Cross-Validation
# Using selected H3 configuration
# -----------------------------------------

from sklearn.model_selection import GroupKFold

cv_features = df[features]
cv_target = df[target]
cv_groups = df["patient_id"]

group_kfold = GroupKFold(n_splits=5)

cv_auc_scores = []
cv_f1_scores = []

print("\n===== 5-FOLD PATIENT-LEVEL CV =====")

for fold, (train_idx, val_idx) in enumerate(
    group_kfold.split(cv_features, cv_target, groups=cv_groups),
    start=1
):

    X_cv_train = cv_features.iloc[train_idx]
    y_cv_train = cv_target.iloc[train_idx]

    X_cv_val = cv_features.iloc[val_idx]
    y_cv_val = cv_target.iloc[val_idx]

    cv_model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(
            (y_cv_train == 0).sum() /
            (y_cv_train == 1).sum()
        ),
        random_state=42,
        n_jobs=-1
    )

    cv_model.fit(X_cv_train, y_cv_train)

    cv_prob = cv_model.predict_proba(X_cv_val)[:, 1]

    # Use the locked threshold from validation tuning
    cv_pred = (cv_prob >= 0.64).astype(int)

    fold_auc = roc_auc_score(y_cv_val, cv_prob)
    fold_f1 = f1_score(y_cv_val, cv_pred)

    cv_auc_scores.append(fold_auc)
    cv_f1_scores.append(fold_f1)

    print(
        f"Fold {fold}: "
        f"ROC-AUC = {fold_auc:.4f}, "
        f"F1 = {fold_f1:.4f}"
    )

print("\n===== CROSS-VALIDATION SUMMARY =====")

print(
    f"Mean ROC-AUC: "
    f"{sum(cv_auc_scores) / len(cv_auc_scores):.4f}"
)

print(
    f"Mean F1: "
    f"{sum(cv_f1_scores) / len(cv_f1_scores):.4f}"
)

print(
    f"ROC-AUC Std: "
    f"{pd.Series(cv_auc_scores).std():.4f}"
)

print(
    f"F1 Std: "
    f"{pd.Series(cv_f1_scores).std():.4f}"
)

# Hyperparameter tuning
# -----------------------------------------

best_model = None
best_config = None
best_threshold = None
best_val_f1 = 0

print("\n===== HYPERPARAMETER TUNING =====")

for config in hyperparameter_configs:

    print(f"\nTraining {config['name']}...")

    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Validation probabilities
    y_val_prob = model.predict_proba(X_val)[:, 1]

    # Find best threshold on validation set
    config_best_threshold = 0
    config_best_f1 = 0

    for threshold in [i / 100 for i in range(10, 91)]:

        y_val_pred = (
            y_val_prob >= threshold
        ).astype(int)

        f1 = f1_score(y_val, y_val_pred)

        if f1 > config_best_f1:
            config_best_f1 = f1
            config_best_threshold = threshold

    print(
        f"{config['name']} | "
        f"Threshold: {config_best_threshold:.2f} | "
        f"Validation F1: {config_best_f1:.4f}"
    )

    # Keep the best model based ONLY on validation F1
    if config_best_f1 > best_val_f1:

        best_val_f1 = config_best_f1
        best_model = model
        best_config = config
        best_threshold = config_best_threshold

print("\n===== BEST MODEL =====")

print("Configuration:")
print(best_config)

print(f"Validation F1: {best_val_f1:.4f}")
print(f"Selected threshold: {best_threshold:.2f}")

# -----------------------------------------
# FINAL TEST EVALUATION
# -----------------------------------------

y_test_prob = best_model.predict_proba(X_test)[:, 1]

y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)

auc = roc_auc_score(y_test, y_test_prob)
f1 = f1_score(y_test, y_test_pred)

print("\n===== FINAL TEST RESULTS =====")
print(f"ROC-AUC: {auc:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_test_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_test_pred))

# -----------------------------------------
# Generate demo patient predictions
# -----------------------------------------

demo_patients = [
    "187a1094-82b0-8949-3e8e-1b78c62581c1",
    "d7af6e65-bec0-a9a9-a7d6-fa39bdfa460d",
    "db55a107-0cb4-0276-8ade-7d4a0a0dd424",
    "327c1aa7-a065-a2e1-4bd7-5c2a1b0ee87a",
    "09f51a16-d166-d7b3-1965-c9f60181446a"
]

demo_df = test_df[
    test_df["patient_id"].isin(demo_patients)
].copy()

demo_prob = best_model.predict_proba(
    demo_df[features]
)[:, 1]

demo_df["predicted_spike_probability"] = demo_prob

demo_df["predicted_spike"] = (
    demo_df["predicted_spike_probability"] >= best_threshold
).astype(int)

# Keep useful dashboard columns
demo_output = demo_df[
    [
        "patient_id",
        "timestamp",
        "glucose_mg_dl",
        "predicted_spike_probability",
        "predicted_spike",
        target
    ]
].copy()

# Rename target for dashboard readability
demo_output = demo_output.rename(
    columns={
        target: "actual_spike_2h"
    }
)

# Save
demo_file = project_root / "data" / "processed" / "demo_predictions.csv"

demo_output.to_csv(
    demo_file,
    index=False
)

print("\n===== DEMO PATIENTS =====")
print("Selected patients:")

for patient in demo_patients:
    print(patient)

print("\nDemo predictions saved to:")
print(demo_file)

# -----------------------------------------
# Feature importance
# -----------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": best_model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance.to_string(index=False))

# Save model
model_dir = project_root / "models"
model_dir.mkdir(parents=True, exist_ok=True)

model_file = model_dir / "xgb_final.pkl"
joblib.dump(best_model, model_file)

print("\nBaseline model saved to:")
print(model_file)

# -----------------------------------------
# Find interesting demo patients
# -----------------------------------------

test_results = test_df.copy()

test_results["predicted_probability"] = (
    best_model.predict_proba(test_df[features])[:, 1]
)

test_results["predicted_spike"] = (
    test_results["predicted_probability"] >= best_threshold
).astype(int)

print("\n===== HIGH-RISK TEST PATIENTS =====")

high_risk = (
    test_results
    .groupby("patient_id")["predicted_probability"]
    .max()
    .sort_values(ascending=False)
    .head(10)
)

print(high_risk)

print("\n===== ACTUAL SPIKE PATIENTS =====")

actual_spikes = (
    test_results[
        test_results[target] == 1
    ]
    .groupby("patient_id")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print(actual_spikes)