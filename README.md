# 🩺 Type 2 Diabetes Digital Twin

### Predicting glucose spike risk 2 hours ahead using synthetic EHR + wearable time-series data

> **Digital Twin Challenge 2026 — Research Proof of Concept**

A patient-level Digital Twin proof of concept that combines simulated EHR information with synthetic continuous glucose monitoring and wearable signals to predict whether glucose will exceed a defined modeling target 2 hours ahead.

The system also uses SHAP explainability to show which features contributed most to each prediction and presents the result through an interactive Streamlit dashboard.

---

## 🎯 Problem

Type 2 diabetes is a dynamic condition in which glucose levels change over time and can be influenced by multiple patient-specific factors.

A static snapshot of a patient's medical record does not capture these temporal changes.

This project explores a Digital Twin approach that combines:

- Historical/static patient information
- Continuous glucose measurements
- Physical activity
- Sleep-stage information
- Machine-learning prediction
- Explainable AI

The goal is to create a continuously updated simulated patient state and forecast glucose-spike risk at a 2-hour horizon.

---

# 🧠 What This Digital Twin Does

For each simulated patient, the system combines:

```text
                 DIGITAL TWIN
                      │
        ┌─────────────┴─────────────┐
        │                           │
   Static EHR                  Dynamic Signals
        │                           │
 Demographics                   Synthetic CGM
 Lab measurements               Steps
 Medications                    Sleep
        │                           │
        └─────────────┬─────────────┘
                      ↓
              Feature Engineering
                      ↓
                 XGBoost Model
                      ↓
             2-Hour Spike Risk
                      ↓
                SHAP Explainability
                      ↓
             Interactive Dashboard