# 🩺 Type 2 Diabetes Digital Twin

### Predicting glucose spike risk 2 hours ahead using synthetic EHR + dynamic wearable signals

> **Digital Twin Challenge 2026 — Research Proof of Concept**

A patient-level Digital Twin proof of concept that combines simulated Electronic Health Record (EHR) information with synthetic continuous glucose monitoring (CGM), activity, and sleep signals to estimate whether glucose will exceed a defined modeling target exactly **2 hours ahead**.

The system combines **data fusion, feature engineering, XGBoost, patient-level validation, SHAP explainability, and an interactive Streamlit dashboard**.

---

## 👤 Team Details

**Team Type:** Individual

**Participant:** Sanchit Kumar Rai

---

## 🎓 College / Incubator Information

**College:** Sri Aurobindo College, University of Delhi

**Incubator:** Not applicable — independently developed project.

---

## 🏷️ Project Title

**Type 2 Diabetes Digital Twin — Explainable 2-Hour Glucose Spike Prediction**

---

## 🎯 Problem Statement

Type 2 diabetes is a dynamic condition in which glucose levels continuously change over time and can be influenced by multiple patient-specific factors.

A static medical record provides historical information but does not represent the patient's changing physiological state.

This project explores a Digital Twin approach that combines:

- Historical patient information
- Laboratory measurements
- Continuous glucose measurements
- Physical activity
- Sleep-stage information
- Machine-learning prediction
- Explainable AI

The objective is to maintain a simulated patient state and forecast whether glucose will exceed a predefined **synthetic modeling target** exactly two hours ahead.

---

## 🏥 Healthcare Use Case

The proposed system demonstrates how a future diabetes Digital Twin could support:

- Patient-specific glucose-risk monitoring
- Early identification of potential glucose excursions
- Integration of historical EHR context with real-time physiological signals
- Explainable prediction rather than black-box risk scores
- Visualization of changing patient state
- Future clinical decision-support research

### Important

This is a **research proof of concept using synthetic data**.

It is **not a clinical decision-support system and must not be used for diagnosis or treatment**.

---

# 🧬 Digital Twin Concept

The Digital Twin represents a simulated patient's evolving state by combining two types of information:

### Static / Historical Context

- Demographics
- Diagnoses
- Laboratory measurements
- Medication information

### Dynamic Signals

- Synthetic CGM
- Step count
- Sleep stage

These signals are fused at the patient/timestamp level and transformed into predictive features.

```text
                 DIGITAL TWIN
                       │
          ┌────────────┴────────────┐
          │                         │
     Static EHR                Dynamic Signals
          │                         │
   Demographics                 Synthetic CGM
   Lab measurements             Steps
   Diagnoses                    Sleep
   Medications
          │                         │
          └────────────┬────────────┘
                       ↓
               Feature Engineering
                       ↓
                  XGBoost Model
                       ↓
             2-Hour Spike Risk
                       ↓
               SHAP Explainability
                       ↓
              Digital Twin Dashboard
