# AirSense AI
## Explainability Engine Specification

---

## Module

utils/06_model_explainability.py

---

## Objective

Develop a professional explainability engine for AirSense AI capable of automatically explaining the selected machine learning model after evaluation.

The module must NOT retrain models.

It must ONLY analyze the best model selected by the evaluation pipeline.

---

## Inputs

The module shall automatically load:

• leaderboard/leaderboard.json
• models/
• datasets/prepared/train.csv

The script must automatically determine the best model from leaderboard.json.

No hardcoded model names.

---

## Model Compatibility

Support the following:

Tree Models

- Decision Tree
- Random Forest
- Extra Trees

using

feature_importances_

--------------------------------

Linear Models

- Linear Regression
- Ridge
- Lasso

using

coef_

--------------------------------

Neural Network

MLP

If intrinsic feature importance is unavailable, generate a warning instead of crashing.

---

## Outputs

Generate

explainability/

feature_importance.csv

top10_features.csv

feature_importance.json

dashboard_explainability.json

--------------------------------

reports/

model_explainability_report.txt

--------------------------------

charts/

feature_importance.png

top10_features.png

pollutant_importance.png

---

## Dashboard JSON

Generate dashboard-ready JSON.

Example

{
    "model":"RandomForest",
    "top_feature":"PM2.5",
    "importance":0.34,
    "summary":"PM2.5 is the strongest AQI driver."
}

---

## Charts

Generate

1.

Complete Feature Importance

Horizontal Bar Chart

----------------------------

2.

Top 10 Features

----------------------------

3.

Pollutant Importance

PM2.5

PM10

NO2

SO2

CO

O3

---

## Report

Generate

MODEL EXPLAINABILITY REPORT

including

Selected Model

Feature Count

Top Features

Least Important Features

Interpretation

Recommendations

Explainability Score

Generation Time

---

## Logging

Professional logging

Example

Loading Leaderboard...

Loading Best Model...

Extracting Feature Importance...

Generating Reports...

Generating Charts...

Saving Dashboard JSON...

Explainability Complete.

---

## Error Handling

Missing leaderboard

↓

Ask user to run

05_model_comparison.py

--------------------------------

Missing model

↓

Ask user to run

04_train_models.py

--------------------------------

No intrinsic feature importance

↓

Generate warning

Continue execution

---

## Code Requirements

PEP8

Type Hints

Professional Docstrings

Professional Logging

Reusable Functions

No duplicated code

Research-backed comments

Robust Exception Handling

Python 3.11 compatible

---

## Libraries

pandas

numpy

matplotlib

joblib

json

logging

pathlib

os

datetime

No SHAP.

No external explainability libraries.

Version 2 enhancement only.