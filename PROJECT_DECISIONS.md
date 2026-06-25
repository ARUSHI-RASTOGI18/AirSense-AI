# ==========================================================
# AIRSENSE AI — ENGINEERING DECISION LOG
# ==========================================================

This document records every major engineering, machine learning, and research decision taken during the development of AirSense AI.

Every important decision must satisfy three conditions:

1. Supported by research literature whenever possible.
2. Practical for hackathon implementation.
3. Explainable to judges during project evaluation.

No preprocessing step, feature engineering step, or AI model will be included simply because it is popular.

Every decision must have a clear scientific and engineering justification.

---

# Decision 001

Topic:
Dataset Selection

Decision:
Use CPCB Air Quality Data in India (2015–2020)

Reason:

• Official CPCB dataset
• Most widely used Indian AQI dataset
• Daily AQI available
• Suitable for regression
• Covers major Indian cities

Status:
APPROVED

---

# Decision 002

Topic:
Dataset Philosophy

Decision:

Historical dataset will NOT be used alone for future forecasting.

Reason:

The historical dataset ends in July 2020.

Forecasting directly to 2026 would create temporal bias.

Future forecasts will combine:

Historical Dataset
+
Live AQI API
+
Live Weather API

Status:
APPROVED

---

# Decision 003

Topic:
COVID-19 Data

Decision:

Do NOT remove COVID observations.

Instead:

Create a binary feature

is_lockdown

Reason:

COVID represents a real-world structural anomaly.

Removing it would destroy valuable information.

Treating it as a separate feature allows the model to distinguish between normal conditions and lockdown conditions.

Status:
APPROVED

---

# Decision 004

Topic:
Duplicate Rows

Decision:

Remove only exact duplicate records.

Reason:

Duplicate sensor readings introduce bias during model training.

Temporal observations must never be removed simply because they occur on consecutive days.

Status:
APPROVED

---

# Decision 005

Topic:
High Missing Columns

Decision:

Remove Xylene.

Reason:

Missing percentage exceeds 60%.

Research consistently recommends removing this feature.

Status:
APPROVED

---

# Decision 006

Topic:
Missing Value Strategy

Decision:

Do NOT delete rows containing missing values.

Primary Imputation Method:

KNN Imputer

Reason:

Research supports advanced imputation.

KNN provides a strong balance between scientific validity, implementation simplicity, explainability, and computational efficiency.

SAITS was considered but rejected because it is unnecessarily complex for hackathon deployment.

Status:
APPROVED

---

# Decision 007

Topic:
Outliers

Decision:

Do NOT remove outliers.

Reason:

Extreme AQI values represent genuine environmental events.

Removing them would reduce the model's ability to learn severe pollution behaviour.

Status:
APPROVED

---

# Decision 008

Topic:
Date Feature Engineering

Decision:

Extract

Year

Month

Day

Weekday

Weekend

Season

Reason:

Seasonality strongly affects AQI.

Working Hour features are rejected because the dataset is daily rather than hourly.

Status:
APPROVED

---

# Decision 009

Topic:
Lag Features

Decision:

Generate AQI lag features.

Initial candidates:

Lag 1

Lag 2

Lag 3

Lag 7

Lag 14

Lag 30

Reason:

Lag variables capture temporal dependency and improve forecasting accuracy.

Status:
APPROVED

---

# Decision 010

Topic:
Rolling Features

Decision:

Generate rolling averages.

Candidates:

3 Day

7 Day

14 Day

30 Day

Reason:

Rolling statistics reduce noise while preserving long-term trends.

Status:
APPROVED

---

# Decision 011

Topic:
Scaling Strategy

Decision:

Scaling will NOT be applied universally.

Scaling depends on the selected AI model.

Examples:

Random Forest
→ No Scaling

XGBoost
→ No Scaling

LightGBM
→ No Scaling

CatBoost
→ No Scaling

Linear Regression
→ Scaling

SVR
→ Scaling

Neural Networks
→ Scaling

Reason:

Different algorithms have different mathematical requirements.

Status:
APPROVED

---

# Decision 012

Topic:
Train-Test Split

Decision:

Use chronological splitting only.

Never use random train_test_split for forecasting.

Reason:

Random splitting causes temporal data leakage.

Status:
APPROVED

---

# Decision 013

Topic:
Model Selection

Decision:

Never rely on a single machine learning algorithm.

Candidate models:

Linear Regression

Random Forest

Extra Trees

XGBoost

LightGBM

CatBoost

(Optional)

LSTM

Reason:

The final production model will be selected using experimental comparison.

Status:
APPROVED

---

# Decision 014

Topic:
Model Evaluation

Decision:

Compare every model using identical datasets and identical train-test splits.

Metrics:

MAE

RMSE

R² Score

MAPE

Training Time

Inference Time

Reason:

Model selection must be evidence-based.

Status:
APPROVED

---

# Decision 015

Topic:
Research Policy

Decision:

Research papers guide engineering decisions.

However,

Research recommendations will never be followed blindly.

Every recommendation will be evaluated against:

• Dataset characteristics
• Computational cost
• Hackathon constraints
• Real-world deployment

Status:
APPROVED

==========================================================
END OF ENGINEERING DECISION LOG
==========================================================