import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Paths
# -------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

synthea_dir = project_root / "synthea" / "output" / "csv"
raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"

processed_dir.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# 1. Identify T2DM patients
# -------------------------------------------------

conditions = pd.read_csv(
    synthea_dir / "conditions.csv"
)

t2dm = conditions[
    conditions["DESCRIPTION"].str.contains(
        "diabetes mellitus type 2",
        case=False,
        na=False
    )
]

t2dm_patients = (
    t2dm["PATIENT"]
    .drop_duplicates()
    .tolist()
)

print(f"T2DM patients: {len(t2dm_patients)}")


# -------------------------------------------------
# 2. Patient demographics
# -------------------------------------------------

patients = pd.read_csv(
    synthea_dir / "patients.csv"
)

patients = patients[
    patients["Id"].isin(t2dm_patients)
].copy()

patients["BIRTHDATE"] = pd.to_datetime(
    patients["BIRTHDATE"]
)

# Calculate age at the end of the synthetic period
reference_date = pd.Timestamp("2026-01-01")

patients["age"] = (
    reference_date.year
    - patients["BIRTHDATE"].dt.year
)

patients["patient_id"] = patients["Id"]

demographics = patients[
    [
        "patient_id",
        "GENDER",
        "age"
    ]
].copy()


# -------------------------------------------------
# 3. Extract clinical observations
# -------------------------------------------------

observations = pd.read_csv(
    synthea_dir / "observations.csv"
)

observations = observations[
    observations["PATIENT"].isin(t2dm_patients)
].copy()


def get_latest_observation(description_pattern, column_name):

    temp = observations[
        observations["DESCRIPTION"].str.contains(
            description_pattern,
            case=False,
            regex=True,
            na=False
        )
    ].copy()

    temp["DATE"] = pd.to_datetime(
        temp["DATE"],
        errors="coerce"
    )

    temp["VALUE_NUMERIC"] = pd.to_numeric(
        temp["VALUE"],
        errors="coerce"
    )

    temp = temp.dropna(
        subset=["VALUE_NUMERIC"]
    )

    temp = temp.sort_values(
        ["PATIENT", "DATE"]
    )

    temp = (
        temp
        .groupby("PATIENT")
        .tail(1)
    )

    temp = temp[
        ["PATIENT", "VALUE_NUMERIC"]
    ].rename(
        columns={
            "PATIENT": "patient_id",
            "VALUE_NUMERIC": column_name
        }
    )

    return temp


# HbA1c
hba1c = get_latest_observation(
    r"Hemoglobin A1c/Hemoglobin\.total in Blood",
    "hba1c"
)

# BMI
bmi = get_latest_observation(
    r"Body mass index \(BMI\) \[Ratio\]",
    "bmi"
)

# Glucose
baseline_glucose = get_latest_observation(
    r"Glucose \[Mass/volume\] in Blood$",
    "baseline_glucose"
)

# Cholesterol
cholesterol = get_latest_observation(
    r"Cholesterol \[Mass/volume\] in Serum or Plasma$",
    "cholesterol"
)

# Triglycerides
triglycerides = get_latest_observation(
    r"Triglyceride \[Mass/volume\] in Serum or Plasma$",
    "triglycerides"
)


# -------------------------------------------------
# 4. Combine static clinical features
# -------------------------------------------------

static_data = demographics.copy()

for feature in [
    hba1c,
    bmi,
    baseline_glucose,
    cholesterol,
    triglycerides
]:

    static_data = static_data.merge(
        feature,
        on="patient_id",
        how="left"
    )


# -------------------------------------------------
# 5. Medication information
# -------------------------------------------------

medications = pd.read_csv(
    synthea_dir / "medications.csv"
)

medications = medications[
    medications["PATIENT"].isin(t2dm_patients)
].copy()

medications_summary = (
    medications
    .groupby("PATIENT")["DESCRIPTION"]
    .apply(
        lambda x: "; ".join(
            sorted(
                set(x.dropna())
            )
        )
    )
    .reset_index()
)

medications_summary.columns = [
    "patient_id",
    "medications"
]

medication_count = (
    medications
    .groupby("PATIENT")["DESCRIPTION"]
    .nunique()
    .reset_index()
)

medication_count.columns = [
    "patient_id",
    "medication_count"
]

static_data = static_data.merge(
    medications_summary,
    on="patient_id",
    how="left"
)

static_data = static_data.merge(
    medication_count,
    on="patient_id",
    how="left"
)


# -------------------------------------------------
# 6. Load wearable data
# -------------------------------------------------

cgm = pd.read_csv(
    raw_dir / "cgm.csv"
)

steps = pd.read_csv(
    raw_dir / "steps.csv"
)

sleep = pd.read_csv(
    raw_dir / "sleep.csv"
)


# -------------------------------------------------
# 7. Merge wearable signals
# -------------------------------------------------

wearable = cgm.merge(
    steps,
    on=["patient_id", "timestamp"],
    how="inner"
)

wearable = wearable.merge(
    sleep,
    on=["patient_id", "timestamp"],
    how="inner"
)


# -------------------------------------------------
# 8. Add static EHR features
# -------------------------------------------------

processed = wearable.merge(
    static_data,
    on="patient_id",
    how="left"
)


# -------------------------------------------------
# 9. Sort data
# -------------------------------------------------

processed["timestamp"] = pd.to_datetime(
    processed["timestamp"]
)

processed = processed.sort_values(
    ["patient_id", "timestamp"]
)


# -------------------------------------------------
# 10. Save processed dataset
# -------------------------------------------------

output_file = (
    processed_dir
    / "t2dm_timeseries.csv"
)

processed.to_csv(
    output_file,
    index=False
)


# -------------------------------------------------
# 11. Validation
# -------------------------------------------------

print("\nMerge completed successfully!")

print(f"Patients: {processed['patient_id'].nunique()}")
print(f"Rows: {len(processed)}")

print("\nColumns:")
print(processed.columns.tolist())

print("\nMissing values:")
print(
    processed.isna()
    .sum()
    .to_string()
)

print("\nSaved to:")
print(output_file)
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

