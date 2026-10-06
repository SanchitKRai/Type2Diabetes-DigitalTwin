import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible random numbers
np.random.seed(42)

# Project paths
project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Load Type 2 Diabetes patient IDs from Synthea
# -------------------------------------------------

patient_file = raw_dir / "t2dm_patients.csv"

patients = pd.read_csv(patient_file)["PATIENT"].dropna().unique()

print(f"T2DM patients found: {len(patients)}")

# -------------------------------------------------
# One day of readings every 5 minutes
# -------------------------------------------------

timestamps = pd.date_range(
    start="2026-01-01 00:00:00",
    end="2026-01-01 23:55:00",
    freq="5min",
)

# Meal times and glucose response
meals = {
    "08:00": 90,
    "13:00": 105,
    "19:00": 95,
}

# Store all patients' data
all_cgm = []
all_steps = []
all_sleep = []

# -------------------------------------------------
# Generate data for every T2DM patient
# -------------------------------------------------

for patient_id in patients:

    # Different baseline for each patient
    baseline = np.random.normal(120, 10)

    glucose = np.full(
        len(timestamps),
        baseline,
        dtype=float
    )

    # -----------------------------
    # Meal-related glucose spikes
    # -----------------------------

    for meal_time, spike_size in meals.items():

        meal_timestamp = pd.Timestamp(
            f"2026-01-01 {meal_time}"
        )

        minutes_after_meal = (
            timestamps - meal_timestamp
        ).total_seconds() / 60

        spike = np.where(
            minutes_after_meal >= 0,
            spike_size
            * (minutes_after_meal / 60)
            * np.exp(-minutes_after_meal / 60),
            0,
        )

        glucose += spike

    # -----------------------------
    # Random glucose noise
    # -----------------------------

    noise = np.random.normal(
        0,
        3,
        len(glucose)
    )

    glucose += noise

    # Prevent unrealistic values
    glucose = np.maximum(
        glucose,
        60
    )

    # -----------------------------
    # CGM dataframe
    # -----------------------------

    cgm_patient = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": timestamps,
        "glucose_mg_dl": glucose.round(1),
    })

    all_cgm.append(cgm_patient)

    # -----------------------------
    # Step-count data
    # -----------------------------

    steps = []

    for timestamp in timestamps:

        hour = timestamp.hour

        if 7 <= hour < 9:
            step_count = np.random.poisson(20)

        elif 12 <= hour < 14:
            step_count = np.random.poisson(15)

        elif 17 <= hour < 20:
            step_count = np.random.poisson(25)

        elif 22 <= hour or hour < 6:
            step_count = np.random.poisson(1)

        else:
            step_count = np.random.poisson(8)

        steps.append(step_count)

    steps_patient = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": timestamps,
        "steps": steps,
    })

    all_steps.append(steps_patient)

    # -----------------------------
    # Sleep-stage data
    # -----------------------------

    sleep_stages = []

    for timestamp in timestamps:

        hour = timestamp.hour

        if hour >= 23 or hour < 7:

            stage = np.random.choice(
                ["LIGHT", "DEEP", "REM"],
                p=[0.55, 0.30, 0.15],
            )

        else:

            stage = "AWAKE"

        sleep_stages.append(stage)

    sleep_patient = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": timestamps,
        "sleep_stage": sleep_stages,
    })

    all_sleep.append(sleep_patient)


# -------------------------------------------------
# Combine all patients
# -------------------------------------------------

cgm_data = pd.concat(
    all_cgm,
    ignore_index=True
)

steps_data = pd.concat(
    all_steps,
    ignore_index=True
)

sleep_data = pd.concat(
    all_sleep,
    ignore_index=True
)


# -------------------------------------------------
# Save raw wearable data
# -------------------------------------------------

cgm_data.to_csv(
    raw_dir / "cgm.csv",
    index=False
)

steps_data.to_csv(
    raw_dir / "steps.csv",
    index=False
)

sleep_data.to_csv(
    raw_dir / "sleep.csv",
    index=False
)


# -------------------------------------------------
# Confirmation
# -------------------------------------------------

print("\nWearable data generated successfully!")

print(f"CGM rows: {len(cgm_data)}")
print(f"Steps rows: {len(steps_data)}")
print(f"Sleep rows: {len(sleep_data)}")

print("\nExpected rows per signal:")
print(f"{len(patients)} patients × {len(timestamps)} readings")
print(f"= {len(patients) * len(timestamps)}")

print("\nFiles saved:")
print(raw_dir / "cgm.csv")
print(raw_dir / "steps.csv")
print(raw_dir / "sleep.csv")