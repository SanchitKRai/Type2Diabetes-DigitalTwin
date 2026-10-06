import pandas as pd
from pathlib import Path

# Project paths
project_root = Path(__file__).resolve().parents[1]
input_file = project_root / "data" / "processed" / "t2dm_timeseries.csv"

# Load data
df = pd.read_csv(input_file)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Make sure each patient's data is ordered chronologically
df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df[["patient_id", "timestamp", "glucose_mg_dl"]].head())
# Glucose change over the previous 15 minutes
df["glucose_change_15min"] = (
    df.groupby("patient_id")["glucose_mg_dl"]
    .diff(periods=3)
)

print("\n15-minute glucose change:")
print(
    df[
        ["patient_id", "timestamp", "glucose_mg_dl",
         "glucose_change_15min"]
    ].head(10).to_string(index=False)
)
# 30-minute rolling average glucose
df["glucose_rolling_mean_30min"] = (
    df.groupby("patient_id")["glucose_mg_dl"]
    .transform(lambda x: x.rolling(window=6).mean())
)

print("\n30-minute rolling mean:")
print(
    df[
        [
            "timestamp",
            "glucose_mg_dl",
            "glucose_change_15min",
            "glucose_rolling_mean_30min"
        ]
    ].head(15).to_string(index=False)
)
# 30-minute glucose variability
df["glucose_rolling_std_30min"] = (
    df.groupby("patient_id")["glucose_mg_dl"]
    .transform(lambda x: x.rolling(window=6).std())
)

print("\n30-minute glucose variability:")
print(
    df[
        [
            "timestamp",
            "glucose_mg_dl",
            "glucose_rolling_mean_30min",
            "glucose_rolling_std_30min"
        ]
    ].head(15).to_string(index=False)
)
# Glucose level 2 hours into the future
df["future_glucose_2h"] = (
    df.groupby("patient_id")["glucose_mg_dl"]
    .shift(-24)
)

print("\nFuture 2-hour glucose:")
print(
    df[
        [
            "timestamp",
            "glucose_mg_dl",
            "future_glucose_2h"
        ]
    ].head(10).to_string(index=False)
)
# Create the 2-hour glucose spike target
df["glucose_spike_2h"] = (
    df["future_glucose_2h"] > 160
).astype(int)

print("\nSpike target distribution:")
print(df["glucose_spike_2h"].value_counts())
print("\nSpike target percentage:")
print(df["glucose_spike_2h"].value_counts(normalize=True) * 100)
# Remove rows where the 2-hour future value is unavailable
df = df.dropna(subset=["future_glucose_2h"])

print("\nAfter removing rows without future data:")
print("Shape:", df.shape)
print("Patients:", df["patient_id"].nunique())
# Time-based features
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute

print("\nTime features:")
print(
    df[
        ["timestamp", "hour", "minute"]
    ].head(15).to_string(index=False)
)
# Steps accumulated over the previous 30 minutes
df["steps_30min"] = (
    df.groupby("patient_id")["steps"]
    .transform(lambda x: x.rolling(window=6).sum())
)

print("\n30-minute activity:")
print(
    df[
        ["timestamp", "steps", "steps_30min"]
    ].head(15).to_string(index=False)
)
# Encode sleep stage as numerical values
sleep_mapping = {
    "AWAKE": 0,
    "LIGHT": 1,
    "DEEP": 2,
    "REM": 3
}

df["sleep_stage_encoded"] = df["sleep_stage"].map(sleep_mapping)

print("\nSleep encoding:")
print(
    df[
        ["timestamp", "sleep_stage", "sleep_stage_encoded"]
    ].head(15).to_string(index=False)
)
print("\nCurrent features:")
print(df.columns.tolist())
# Encode gender
df["gender_encoded"] = df["GENDER"].map({
    "M": 0,
    "F": 1
})

print("\nGender encoding:")
print(df[["GENDER", "gender_encoded"]].drop_duplicates().to_string(index=False))
# Features that will be used by the ML model
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
    "minute"
]

# Remove rows with missing model features
df = df.dropna(subset=model_features)

print("\nAfter final feature cleaning:")
print("Shape:", df.shape)
print("Missing values:")
print(df[model_features].isna().sum().sum())
# Save final ML-ready dataset
output_file = project_root / "data" / "processed" / "ml_ready.csv"

df.to_csv(output_file, index=False)

print("\nML-ready dataset saved!")
print("File:", output_file)
print("Final shape:", df.shape)