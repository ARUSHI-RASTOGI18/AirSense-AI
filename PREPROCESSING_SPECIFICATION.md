# AirSense AI - Data Preprocessing Specification

## Purpose

This document defines the exact responsibilities of the preprocessing pipeline.

The preprocessing script must only perform dataset cleaning and validation.

It must NOT perform feature engineering, model training, scaling, encoding, or prediction.

---

# Input

datasets/raw/city_day.csv

---

# Output

datasets/processed/clean_city_day.csv

datasets/reports/preprocessing_report.txt

---

# Objectives

The preprocessing pipeline should:

- Load the raw dataset.
- Validate dataset structure.
- Verify required columns.
- Remove exact duplicate rows.
- Remove high-missing-value columns approved by the engineering team.
- Convert Date into datetime format.
- Sort data chronologically for every city.
- Create basic calendar features.
- Create COVID lockdown indicator.
- Save cleaned dataset.
- Generate preprocessing report.

---

# Responsibilities

## 1. Dataset Validation

The script must verify that all required columns exist before processing.

If any required column is missing, execution must stop with a meaningful error.

---

## 2. Duplicate Handling

Remove only exact duplicate rows.

Never remove legitimate observations from different dates.

---

## 3. Column Removal

Approved for removal:

- Xylene

No other feature may be removed without approval.

---

## 4. Date Processing

Convert Date into datetime format.

Create:

- Year
- Month
- Day
- DayOfWeek
- IsWeekend
- Season

---

## 5. COVID Feature

Create

is_lockdown

Rule:

March 2020 to July 2020 → 1

Otherwise → 0

---

## 6. Sorting

Sort dataset by

City

then

Date

---

## 7. Reporting

Generate preprocessing_report.txt

Include:

- Initial rows
- Final rows
- Duplicate rows removed
- Columns removed
- New columns created
- Execution time

---

# Out of Scope

The preprocessing pipeline MUST NOT:

- Fill missing values
- Scale data
- Encode categorical variables
- Create lag features
- Create rolling statistics
- Train machine learning models
- Generate predictions

These tasks belong to later pipeline stages.

---

# Engineering Principles

- Never modify the original dataset.
- Never overwrite raw files.
- Keep the code modular.
- Keep functions small.
- Handle errors gracefully.
- Produce reproducible outputs.
- Follow PEP8.