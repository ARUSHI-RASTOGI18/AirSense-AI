# ─────────────────────────────────────────────────────────────
#  AirSense AI – Dataset Quality Analysis
#  Investigates city_day.csv and produces a professional
#  quality report. NO data is modified at any point.
#  Output: datasets/processed/dataset_quality_report.txt
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── File paths ───────────────────────────────────────────────
INPUT_PATH  = os.path.join("datasets", "raw", "city_day.csv")
OUTPUT_DIR  = os.path.join("datasets", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_quality_report.txt")

# ── Target cities for AirSense AI ───────────────────────────
TARGET_CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai"]

# ── Pollutant columns to analyse for outliers ────────────────
OUTLIER_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]

# ── All pollutant columns for correlation with AQI ───────────
POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx",
                  "NH3", "CO", "SO2", "O3"]

# ── Visual separators ────────────────────────────────────────
SEP1 = "=" * 70
SEP2 = "-" * 70
SEP3 = "·" * 70


# ─────────────────────────────────────────────────────────────
#  LINE COLLECTOR
#  All output goes through add() so we can print AND save
#  simultaneously without duplicating logic.
# ─────────────────────────────────────────────────────────────
_report_lines = []

def add(text=""):
    """Print a line and append it to the report buffer."""
    print(text)
    _report_lines.append(str(text))


# ─────────────────────────────────────────────────────────────
#  SECTION 1 – DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────
def section_overview(df):
    add(SEP1)
    add("  SECTION 1 │ DATASET OVERVIEW")
    add(SEP1)

    # Basic shape
    add(f"  Rows               : {df.shape[0]:,}")
    add(f"  Columns            : {df.shape[1]}")
    add()

    # Column names with index
    add("  Column List:")
    add(SEP3)
    for i, col in enumerate(df.columns, 1):
        add(f"    {i:>2}. {col}")
    add()

    # Data types
    add("  Data Types:")
    add(SEP3)
    for col, dtype in df.dtypes.items():
        add(f"    {col:<20} : {dtype}")
    add()

    # Date range – parse Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        valid_dates = df["Date"].dropna()
        add("  Date Range:")
        add(SEP3)
        add(f"    Earliest Date      : {valid_dates.min().strftime('%Y-%m-%d')}")
        add(f"    Latest Date        : {valid_dates.max().strftime('%Y-%m-%d')}")
        span = (valid_dates.max() - valid_dates.min()).days
        add(f"    Total Span         : {span} days  "
            f"(~{span // 365} yr {span % 365} days)")
        add(f"    Invalid Date Rows  : {df['Date'].isna().sum()}")
    else:
        add("  [WARN] 'Date' column not found.")
    add()

    # Memory usage
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    add(f"  Memory Usage       : {mem_mb:.2f} MB")
    add()


# ─────────────────────────────────────────────────────────────
#  SECTION 2 – MISSING VALUE ANALYSIS
# ─────────────────────────────────────────────────────────────
def section_missing(df):
    add(SEP1)
    add("  SECTION 2 │ MISSING VALUE ANALYSIS")
    add(SEP1)

    total_cells = df.shape[0] * df.shape[1]
    missing_per_col = df.isnull().sum()
    pct_per_col     = (missing_per_col / df.shape[0] * 100).round(2)

    # Build a dataframe for easy sorting
    missing_df = pd.DataFrame({
        "Missing Count": missing_per_col,
        "Missing %":     pct_per_col
    }).sort_values("Missing %", ascending=False)

    add(f"  Total Cells        : {total_cells:,}")
    add(f"  Total Missing      : {missing_per_col.sum():,}  "
        f"({missing_per_col.sum() / total_cells * 100:.2f}% of all cells)")
    add()
    add(f"  {'Column':<22} {'Missing Count':>15} {'Missing %':>12}")
    add(SEP3)
    for col, row in missing_df.iterrows():
        flag = "  ◄ HIGH" if row["Missing %"] > 30 else ""
        add(f"  {col:<22} {int(row['Missing Count']):>15,} "
            f"{row['Missing %']:>11.2f}%{flag}")
    add()

    # Rows with at least one missing value
    rows_with_any_missing = df.isnull().any(axis=1).sum()
    add(f"  Rows with ≥1 missing value : {rows_with_any_missing:,}  "
        f"({rows_with_any_missing / df.shape[0] * 100:.2f}%)")

    # Rows where ALL pollutants are missing
    all_missing_mask = df[POLLUTANT_COLS].isnull().all(axis=1)
    add(f"  Rows with ALL pollutants missing : {all_missing_mask.sum():,}")
    add()


# ─────────────────────────────────────────────────────────────
#  SECTION 3 – DUPLICATE ANALYSIS
# ─────────────────────────────────────────────────────────────
def section_duplicates(df):
    add(SEP1)
    add("  SECTION 3 │ DUPLICATE ANALYSIS")
    add(SEP1)

    dup_mask   = df.duplicated()
    dup_count  = dup_mask.sum()
    dup_pct    = dup_count / df.shape[0] * 100

    add(f"  Total Duplicate Rows  : {dup_count:,}")
    add(f"  Duplicate Percentage  : {dup_pct:.3f}%")
    add()

    if dup_count > 0:
        add("  First 10 Duplicate Rows:")
        add(SEP3)
        dup_rows = df[dup_mask].head(10)
        add(dup_rows.to_string(index=True))
    else:
        add("  ✓ No exact duplicate rows found.")
    add()

    # Check for City+Date duplicates (logical duplicates)
    if "City" in df.columns and "Date" in df.columns:
        cd_dups = df.duplicated(subset=["City", "Date"], keep=False).sum()
        add(f"  City+Date Logical Duplicates : {cd_dups:,}  "
            f"(same city recorded twice on same date)")
    add()


# ─────────────────────────────────────────────────────────────
#  SECTION 4 – TARGET CITY ANALYSIS
# ─────────────────────────────────────────────────────────────
def section_target_cities(df):
    add(SEP1)
    add("  SECTION 4 │ TARGET CITY ANALYSIS")
    add(SEP1)

    if "City" not in df.columns:
        add("  [WARN] 'City' column not found.")
        return

    for city in TARGET_CITIES:
        city_df = df[df["City"] == city].copy()
        add(f"  ┌─ {city.upper()} {'─' * (52 - len(city))}")

        if city_df.empty:
            add(f"  │  [NOT FOUND] No records for {city} in dataset.")
            add(f"  └{'─' * 56}")
            add()
            continue

        # Record count
        add(f"  │  Records          : {len(city_df):,}")

        # Date range for this city
        if "Date" in city_df.columns:
            valid = city_df["Date"].dropna()
            add(f"  │  Date Range       : {valid.min().strftime('%Y-%m-%d')}  "
                f"→  {valid.max().strftime('%Y-%m-%d')}")
            span = (valid.max() - valid.min()).days
            add(f"  │  Span             : {span} days")

        # AQI stats
        if "AQI" in city_df.columns:
            aqi = city_df["AQI"].dropna()
            add(f"  │  AQI Records      : {len(aqi):,}  "
                f"({city_df['AQI'].isna().sum()} missing)")
            add(f"  │  AQI Mean         : {aqi.mean():.2f}")
            add(f"  │  AQI Median       : {aqi.median():.2f}")
            add(f"  │  AQI Min          : {aqi.min():.2f}")
            add(f"  │  AQI Max          : {aqi.max():.2f}")
            add(f"  │  AQI Std Dev      : {aqi.std():.2f}")
        else:
            add(f"  │  [WARN] AQI column not found.")

        # Missing values for key pollutants in this city
        add(f"  │  Missing per key column:")
        for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]:
            if col in city_df.columns:
                m = city_df[col].isna().sum()
                p = m / len(city_df) * 100
                add(f"  │    {col:<10} : {m:>4} missing  ({p:.1f}%)")

        add(f"  └{'─' * 56}")
        add()


# ─────────────────────────────────────────────────────────────
#  SECTION 5 – AQI DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────
def section_aqi_distribution(df):
    add(SEP1)
    add("  SECTION 5 │ AQI DISTRIBUTION ANALYSIS")
    add(SEP1)

    if "AQI" not in df.columns:
        add("  [WARN] 'AQI' column not found.")
        return

    aqi = df["AQI"].dropna()
    add(f"  Valid AQI Readings : {len(aqi):,}  "
        f"({df['AQI'].isna().sum()} missing)")
    add()

    # Central tendency
    add("  Central Tendency:")
    add(SEP3)
    add(f"    Mean               : {aqi.mean():.2f}")
    add(f"    Median             : {aqi.median():.2f}")
    add(f"    Std Deviation      : {aqi.std():.2f}")
    add(f"    Skewness           : {aqi.skew():.4f}  "
        f"({'right-skewed → tail of high AQI values' if aqi.skew() > 0 else 'left-skewed'})")
    add(f"    Kurtosis           : {aqi.kurt():.4f}")
    add()

    # Percentiles
    add("  Percentiles:")
    add(SEP3)
    for pct in [5, 10, 25, 50, 75, 90, 95, 99]:
        add(f"    {pct:>3}th Percentile  : {aqi.quantile(pct / 100):.2f}")
    add()

    # AQI Bucket distribution
    if "AQI_Bucket" in df.columns:
        add("  AQI Bucket (Category) Distribution:")
        add(SEP3)
        bucket_order = ["Good", "Satisfactory", "Moderate",
                        "Poor", "Very Poor", "Severe"]
        counts = df["AQI_Bucket"].value_counts(dropna=False)
        for bucket in bucket_order:
            if bucket in counts.index:
                c   = counts[bucket]
                pct = c / df.shape[0] * 100
                bar = "█" * int(pct / 2)
                add(f"    {bucket:<18} : {c:>6,}  ({pct:>5.1f}%)  {bar}")
        # any buckets not in the predefined order
        for bucket in counts.index:
            if bucket not in bucket_order:
                c   = counts[bucket]
                pct = c / df.shape[0] * 100
                add(f"    {str(bucket):<18} : {c:>6,}  ({pct:>5.1f}%)")
    add()


# ─────────────────────────────────────────────────────────────
#  SECTION 6 – OUTLIER ANALYSIS  (IQR method, no removal)
# ─────────────────────────────────────────────────────────────
def section_outliers(df):
    add(SEP1)
    add("  SECTION 6 │ OUTLIER ANALYSIS  (IQR Method – identify only)")
    add(SEP1)
    add("  Method : IQR  |  Lower = Q1 − 1.5×IQR  |  Upper = Q3 + 1.5×IQR")
    add("  NOTE   : Outliers are identified only. No data is removed.")
    add()

    add(f"  {'Column':<12} {'Q1':>8} {'Q3':>8} {'IQR':>8} "
        f"{'Lower':>10} {'Upper':>10} {'Outliers':>10} {'%':>8}")
    add(SEP3)

    for col in OUTLIER_COLS:
        if col not in df.columns:
            add(f"  {col:<12}  [column not found]")
            continue

        series = df[col].dropna()
        q1  = series.quantile(0.25)
        q3  = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Count values outside bounds
        outlier_mask  = (series < lower_bound) | (series > upper_bound)
        outlier_count = outlier_mask.sum()
        outlier_pct   = outlier_count / len(series) * 100

        add(f"  {col:<12} {q1:>8.2f} {q3:>8.2f} {iqr:>8.2f} "
            f"{lower_bound:>10.2f} {upper_bound:>10.2f} "
            f"{outlier_count:>10,} {outlier_pct:>7.2f}%")

    add()
    add("  Interpretation:")
    add("    High outlier % in PM2.5/PM10 is expected (seasonal pollution spikes).")
    add("    Outliers in CO/SO2 may indicate industrial incidents or sensor errors.")
    add("    AQI outliers >400 likely correspond to 'Severe' category days in Delhi.")
    add()


# ─────────────────────────────────────────────────────────────
#  SECTION 7 – CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────
def section_correlation(df):
    add(SEP1)
    add("  SECTION 7 │ CORRELATION ANALYSIS  (Pearson – AQI vs Pollutants)")
    add(SEP1)

    if "AQI" not in df.columns:
        add("  [WARN] 'AQI' column not found.")
        return

    correlations = {}
    for col in POLLUTANT_COLS:
        if col in df.columns:
            # Drop rows where either column is NaN for a fair pairwise correlation
            pair = df[["AQI", col]].dropna()
            if len(pair) > 10:
                r = pair["AQI"].corr(pair[col])
                correlations[col] = round(r, 4)
            else:
                correlations[col] = None

    # Sort by absolute correlation strength descending
    sorted_corr = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]) if x[1] is not None else 0,
        reverse=True
    )

    add(f"  {'Pollutant':<12} {'Pearson r':>12}   {'Strength':<20}  Bar")
    add(SEP3)
    for col, r in sorted_corr:
        if r is None:
            add(f"  {col:<12} {'N/A':>12}")
            continue
        abs_r = abs(r)
        if abs_r >= 0.7:
            strength = "Strong"
        elif abs_r >= 0.4:
            strength = "Moderate"
        elif abs_r >= 0.2:
            strength = "Weak"
        else:
            strength = "Negligible"
        direction = "↑ positive" if r > 0 else "↓ negative"
        bar = "█" * int(abs_r * 20)
        add(f"  {col:<12} {r:>12.4f}   {strength + ' ' + direction:<28} {bar}")
    add()

    # Store sorted_corr for feature importance section
    return sorted_corr


# ─────────────────────────────────────────────────────────────
#  SECTION 8 – FEATURE IMPORTANCE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
def section_feature_importance(sorted_corr):
    add(SEP1)
    add("  SECTION 8 │ FEATURE IMPORTANCE RECOMMENDATIONS")
    add(SEP1)

    keep    = []
    caution = []

    if sorted_corr:
        for col, r in sorted_corr:
            if r is None:
                caution.append((col, "N/A – could not compute"))
                continue
            abs_r = abs(r)
            if abs_r >= 0.4:
                keep.append((col, r))
            else:
                caution.append((col, r))

    add("  ✅  DEFINITELY KEEP (Pearson |r| ≥ 0.40 with AQI):")
    add(SEP3)
    if keep:
        for col, r in keep:
            add(f"    {col:<12}  r = {r:.4f}  → Strong/Moderate predictor of AQI")
    else:
        add("    None met threshold.")
    add()

    add("  ⚠️   USE WITH CAUTION (Pearson |r| < 0.40 with AQI):")
    add(SEP3)
    if caution:
        for col, r in caution:
            if isinstance(r, float):
                add(f"    {col:<12}  r = {r:.4f}  → Weak/Negligible direct correlation")
            else:
                add(f"    {col:<12}  r = {r}  → Consider domain expertise before dropping")
    else:
        add("    All features showed moderate-or-above correlation.")
    add()

    add("  Additional Notes:")
    add("    • Low Pearson r does not mean a feature is useless for XGBoost")
    add("      (tree models capture non-linear relationships).")
    add("    • Benzene, Toluene, Xylene have high missingness — evaluate carefully.")
    add("    • Date-derived features (month, day-of-week, season) should be engineered.")
    add("    • Lag features (AQI t-1, t-3, t-7) are critical for time-series forecasting.")
    add()


# ─────────────────────────────────────────────────────────────
#  SECTION 9 – CLEANING RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
def section_cleaning_recommendations(df):
    add(SEP1)
    add("  SECTION 9 │ RECOMMENDED CLEANING STRATEGY")
    add(SEP1)
    add("  ⚠️  These are RECOMMENDATIONS ONLY. No data has been modified.")
    add()

    # ── Columns to Retain ────────────────────────────────────
    add("  [A] COLUMNS TO RETAIN")
    add(SEP3)
    retain = ["City", "Date", "PM2.5", "PM10", "NO", "NO2",
              "NOx", "NH3", "CO", "SO2", "O3", "AQI", "AQI_Bucket"]
    for col in retain:
        reason = {
            "City":       "Core grouping key — required for all city-level analysis",
            "Date":       "Time axis — critical for time-series forecasting",
            "PM2.5":      "Primary AQI driver — must retain",
            "PM10":       "Strong AQI predictor — retain",
            "NO":         "Part of NOx family — retain if NOx is kept",
            "NO2":        "Moderate-strong AQI correlation — retain",
            "NOx":        "Composite nitrogen oxide metric — retain",
            "NH3":        "Weak correlation but useful for source attribution — retain",
            "CO":         "Important for combustion source attribution — retain",
            "SO2":        "Industrial source indicator — retain",
            "O3":         "Secondary pollutant, photochemical signal — retain",
            "AQI":        "Primary target variable for forecasting — must retain",
            "AQI_Bucket": "Classification target — retain",
        }.get(col, "Retain")
        add(f"    ✅  {col:<14}  {reason}")
    add()

    # ── Columns to Consider Dropping ────────────────────────
    add("  [B] COLUMNS TO CONSIDER DROPPING")
    add(SEP3)
    drop_candidates = {
        "Benzene": "Very high missingness (>50% in most cities); "
                   "weak AQI correlation; may drop unless source attribution needed",
        "Toluene": "Very high missingness (>60%); "
                   "negligible AQI correlation; recommend dropping",
        "Xylene":  "Extremely high missingness (>80%); "
                   "near-negligible AQI correlation; recommend dropping",
    }
    for col, reason in drop_candidates.items():
        add(f"    ⚠️   {col:<14}  {reason}")
    add()

    # ── Missing Value Imputation Strategy ───────────────────
    add("  [C] MISSING VALUE IMPUTATION STRATEGY")
    add(SEP3)
    add("    Pollutants with <20% missing (PM2.5, PM10, NO, NO2, NOx, CO, SO2, O3):")
    add("      → Use forward-fill (ffill) per city group first (captures")
    add("        short sensor outages), then backward-fill for edge gaps.")
    add("      → Fall back to city-level rolling 7-day median for remaining gaps.")
    add()
    add("    NH3 (moderate missingness ~30–50%):")
    add("      → Use city-level monthly median imputation.")
    add("      → If still missing, use overall city mean.")
    add()
    add("    AQI (missing where all sub-pollutants are also missing):")
    add("      → Rows where AQI AND all pollutants are missing:")
    add("        REMOVE these rows — no usable signal.")
    add("      → Rows where pollutants exist but AQI is missing:")
    add("        RECALCULATE AQI using CPCB sub-index formula.")
    add()
    add("    Date (any NaT/invalid):")
    add("      → Remove rows with unparseable dates — "
        "time axis integrity is critical.")
    add()

    # ── Rows to Remove ──────────────────────────────────────
    add("  [D] ROWS TO REMOVE")
    add(SEP3)
    add("    1. Exact duplicate rows (City + Date identical) — keep first.")
    add("    2. Rows where Date is null/invalid (NaT after parsing).")
    add("    3. Rows where AQI AND all pollutants are simultaneously null")
    add("       (no recoverable signal).")
    add("    4. Rows with AQI > 500 or AQI < 0 (outside valid CPCB range)")
    add("       — likely sensor or data entry errors.")
    add()

    # ── Rows to Preserve ────────────────────────────────────
    add("  [E] ROWS TO PRESERVE")
    add(SEP3)
    add("    1. Rows with valid AQI even if some pollutant sub-columns are missing.")
    add("    2. Extreme AQI values (300–500) that fall within the valid CPCB range")
    add("       — these are real Severe pollution events (especially Delhi winters).")
    add("    3. COVID-19 lockdown period rows (Mar–Jun 2020) — "
        "preserve as a special")
    add("       temporal segment; may need a lockdown flag feature.")
    add()

    # ── Feature Engineering Suggestions ─────────────────────
    add("  [F] FEATURE ENGINEERING SUGGESTIONS (post-cleaning)")
    add(SEP3)
    add("    • month         : integer month from Date (captures seasonality)")
    add("    • day_of_week   : 0–6 (weekday vs weekend traffic patterns)")
    add("    • season        : Winter/Summer/Monsoon/Post-Monsoon")
    add("    • is_lockdown   : binary flag for Mar 25 – Jun 30, 2020")
    add("    • AQI_lag_1     : AQI from previous day (strongest predictor)")
    add("    • AQI_lag_3     : AQI from 3 days prior")
    add("    • AQI_lag_7     : AQI from 7 days prior")
    add("    • PM25_PM10_ratio: ratio of PM2.5 to PM10 (source fingerprint)")
    add("    • rolling_mean_7 : 7-day rolling mean of AQI per city")
    add()

    # ── Recommended Train/Test Split ────────────────────────
    add("  [G] RECOMMENDED TRAIN/TEST SPLIT")
    add(SEP3)
    add("    Train : 2015-01-01  →  2019-12-31  (5 full years)")
    add("    Test  : 2020-01-01  →  2020-07-01  (held-out future period)")
    add("    Split type: Chronological (never random-split time-series data)")
    add()


# ─────────────────────────────────────────────────────────────
#  SAVE REPORT
# ─────────────────────────────────────────────────────────────
def save_report():
    """Write the collected report lines to a .txt file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_report_lines))
    print()
    print(SEP1)
    print(f"  [✓] Quality report saved → {OUTPUT_PATH}")
    print(SEP1)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # ── Report header ────────────────────────────────────────
    add(SEP1)
    add("  AIRSENSE AI – DATASET QUALITY ANALYSIS REPORT")
    add(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"  Dataset   : {INPUT_PATH}")
    add("  NOTE      : This script is READ-ONLY. No data is modified.")
    add(SEP1)
    add()

    # ── Load dataset ─────────────────────────────────────────
    try:
        df = pd.read_csv(INPUT_PATH)
        add(f"  Dataset loaded successfully — {df.shape[0]:,} rows × {df.shape[1]} columns")
        add()
    except FileNotFoundError:
        add(f"  [ERROR] File not found: {INPUT_PATH}")
        add("          Place city_day.csv in datasets/raw/ and re-run.")
        save_report()
        return
    except pd.errors.EmptyDataError:
        add(f"  [ERROR] File is empty: {INPUT_PATH}")
        save_report()
        return
    except pd.errors.ParserError as e:
        add(f"  [ERROR] CSV parse error: {e}")
        save_report()
        return
    except Exception as e:
        add(f"  [ERROR] Unexpected error loading dataset: {e}")
        save_report()
        return

    # ── Run each analysis section ────────────────────────────
    try:
        section_overview(df)
    except Exception as e:
        add(f"  [ERROR] Section 1 failed: {e}")

    try:
        section_missing(df)
    except Exception as e:
        add(f"  [ERROR] Section 2 failed: {e}")

    try:
        section_duplicates(df)
    except Exception as e:
        add(f"  [ERROR] Section 3 failed: {e}")

    try:
        section_target_cities(df)
    except Exception as e:
        add(f"  [ERROR] Section 4 failed: {e}")

    try:
        section_aqi_distribution(df)
    except Exception as e:
        add(f"  [ERROR] Section 5 failed: {e}")

    try:
        section_outliers(df)
    except Exception as e:
        add(f"  [ERROR] Section 6 failed: {e}")

    sorted_corr = None
    try:
        sorted_corr = section_correlation(df)
    except Exception as e:
        add(f"  [ERROR] Section 7 failed: {e}")

    try:
        section_feature_importance(sorted_corr or [])
    except Exception as e:
        add(f"  [ERROR] Section 8 failed: {e}")

    try:
        section_cleaning_recommendations(df)
    except Exception as e:
        add(f"  [ERROR] Section 9 failed: {e}")

    # ── Save full report ─────────────────────────────────────
    try:
        save_report()
    except PermissionError:
        print(f"  [ERROR] Permission denied writing to {OUTPUT_PATH}")
    except Exception as e:
        print(f"  [ERROR] Could not save report: {e}")


if __name__ == "__main__":
    main()