# 30-Day Readmission Risk Tutorial with MIMIC-III

A beginner-friendly machine learning tutorial that predicts whether an adult patient has another hospital admission within 30 days after discharge. The workflow uses structured MIMIC-III admission-history data, compares logistic regression with random forest, and emphasizes leakage prevention, patient-disjoint evaluation, and responsible interpretation.

## What this tutorial teaches

- How to create a 30-day readmission label from structured admissions data.
- How to use admission-time and prior-history features without leaking future information.
- Why train/test data should be split by patient rather than by admission.
- How to compare logistic regression and random forest using AUROC and average precision.
- Why predictive performance does not establish clinical usefulness, fairness, or causality.

## Data access and privacy

This repository contains **code only**. It does not include MIMIC-III data, patient-level outputs, credentials, clinical notes, or generated figures. Running this tutorial requires authorized access to the full MIMIC-III database and compliance with its data-use agreement. Do not upload MIMIC data or patient-level results to a public repository.

## Setup

1. Create a Python 3.10+ environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain authorized access to the full MIMIC-III database.
4. In `readmission_risk_tutorial.py`, set `MIMIC_ROOT` to the directory containing `ADMISSIONS.csv.gz` and `PATIENTS.csv.gz`.
5. Run the tutorial:

   ```bash
   python readmission_risk_tutorial.py
   ```

The script creates an `outputs/` folder containing aggregate metrics and an ROC curve. It does not export raw data or patient-level predictions.

## Reproducibility

The script fixes the random seed at 42. It uses a patient-disjoint 80/20 split, median imputation and scaling for numeric features, one-hot encoding for categorical features, logistic regression with balanced class weights, and a 300-tree random forest.

## Important limitation

This is an educational baseline, not a clinical decision-support system. External validation, calibration assessment, fairness analysis, clinician review, and prospective evaluation are needed before any clinical application.

## Citation

Johnson AEW, et al. MIMIC-III, a freely accessible critical care database. *Scientific Data*. 2016;3:160035. doi:10.1038/sdata.2016.35
