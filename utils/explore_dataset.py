# ─────────────────────────────────────────────────────────────
#  AirSense AI – Dataset Explorer
#  Loads city_day.csv and prints a full diagnostic summary.
#  Saves a text report to datasets/processed/dataset_summary.txt
# ─────────────────────────────────────────────────────────────

import pandas as pd
import os
from datetime import datetime

# ── File paths ───────────────────────────────────────────────
INPUT_PATH  = os.path.join("datasets", "raw", "city_day.csv")
OUTPUT_DIR  = os.path.join("datasets", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_summary.txt")

# ── Target cities for AirSense AI ───────────────────────────
TARGET_CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai"]

# ── Separator helper for clean console output ────────────────
SEP  = "=" * 60
SEP2 = "-" * 60

def build_summary(df):
    """
    Build the full diagnostic summary as a list of strings.
    Returns lines so they can be both printed and saved to file.
    """
    lines = []

    def add(text=""):
        """Append a line and print it immediately."""
        lines.append(text)
        print(text)

    # ── Header ───────────────────────────────────────────────
    add(SEP)
    add("  AirSense AI – Dataset Exploration Report")
    add(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"  Source    : {INPUT_PATH}")
    add(SEP)

    # ── 1. Basic Shape ───────────────────────────────────────
    add("\n[1] BASIC SHAPE")
    add(SEP2)
    add(f"  Total Rows    : {df.shape[0]:,}")
    add(f"  Total Columns : {df.shape[1]}")

    # ── 2. Column Names ──────────────────────────────────────
    add("\n[2] COLUMN NAMES")
    add(SEP2)
    for i, col in enumerate(df.columns, 1):
        add(f"  {i:>2}. {col}")

    # ── 3. Data Types ────────────────────────────────────────
    add("\n[3] DATA TYPES")
    add(SEP2)
    for col, dtype in df.dtypes.items():
        add(f"  {col:<20} : {dtype}")

    # ── 4. Missing Values per Column ─────────────────────────
    add("\n[4] MISSING VALUES PER COLUMN")
    add(SEP2)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for col in df.columns:
        add(f"  {col:<20} : {missing[col]:>6,} missing  ({missing_pct[col]:>6.2f}%)")
    add(f"\n  Total missing cells : {missing.sum():,}")

    # ── 5. Date Range ────────────────────────────────────────
    add("\n[5] DATE RANGE")
    add(SEP2)
    if "Date" in df.columns:
        # Convert Date column to datetime if not already
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        earliest = df["Date"].min()
        latest   = df["Date"].max()
        span     = (latest - earliest).days
        add(f"  Earliest Date : {earliest.strftime('%Y-%m-%d')}")
        add(f"  Latest Date   : {latest.strftime('%Y-%m-%d')}")
        add(f"  Total Span    : {span} days  (~{span//365} years {span%365} days)")
    else:
        add("  [WARN] 'Date' column not found in dataset.")

    # ── 6. City Coverage ─────────────────────────────────────
    add("\n[6] CITY COVERAGE")
    add(SEP2)
    if "City" in df.columns:
        unique_cities = df["City"].dropna().unique()
        add(f"  Total Unique Cities : {len(unique_cities)}")
        add("\n  First 20 Cities:")
        for i, city in enumerate(sorted(unique_cities)[:20], 1):
            add(f"    {i:>2}. {city}")
    else:
        add("  [WARN] 'City' column not found in dataset.")

    # ── 7. Records per Target City ───────────────────────────
    add("\n[7] RECORDS PER TARGET CITY")
    add(SEP2)
    if "City" in df.columns:
        for city in TARGET_CITIES:
            city_df = df[df["City"] == city]
            count   = len(city_df)
            if count > 0:
                # Date range specific to this city
                c_min = city_df["Date"].min().strftime("%Y-%m-%d") if "Date" in df.columns else "N/A"
                c_max = city_df["Date"].max().strftime("%Y-%m-%d") if "Date" in df.columns else "N/A"
                add(f"  {city:<15} : {count:>5,} records   [{c_min}  →  {c_max}]")
            else:
                add(f"  {city:<15} : 0 records  [NOT FOUND IN DATASET]")
    else:
        add("  [WARN] 'City' column not found in dataset.")

    # ── 8. AQI Statistics ────────────────────────────────────
    add("\n[8] AQI STATISTICS")
    add(SEP2)
    if "AQI" in df.columns:
        aqi_series = df["AQI"].dropna()
        add(f"  Total AQI readings : {len(aqi_series):,}")
        add(f"  Mean AQI           : {aqi_series.mean():.2f}")
        add(f"  Median AQI         : {aqi_series.median():.2f}")
        add(f"  Std Deviation      : {aqi_series.std():.2f}")
        add(f"  Min AQI            : {aqi_series.min():.2f}")
        add(f"  Max AQI            : {aqi_series.max():.2f}")
        add(f"  25th Percentile    : {aqi_series.quantile(0.25):.2f}")
        add(f"  75th Percentile    : {aqi_series.quantile(0.75):.2f}")

        # AQI bucket breakdown if column exists
        if "AQI_Bucket" in df.columns:
            add("\n  AQI Category Breakdown:")
            bucket_counts = df["AQI_Bucket"].value_counts(dropna=False)
            for bucket, count in bucket_counts.items():
                pct = count / len(df) * 100
                add(f"    {str(bucket):<20} : {count:>6,}  ({pct:.1f}%)")

        # Per-city AQI summary for target cities
        add("\n  Mean AQI per Target City:")
        if "City" in df.columns:
            for city in TARGET_CITIES:
                city_aqi = df[df["City"] == city]["AQI"].dropna()
                if len(city_aqi) > 0:
                    add(f"    {city:<15} : Mean={city_aqi.mean():.1f}  "
                        f"Min={city_aqi.min():.0f}  Max={city_aqi.max():.0f}")
                else:
                    add(f"    {city:<15} : No AQI data")
    else:
        add("  [WARN] 'AQI' column not found in dataset.")

    # ── Footer ───────────────────────────────────────────────
    add("\n" + SEP)
    add("  AirSense AI – Exploration Complete")
    add(SEP)

    return lines


def save_report(lines):
    """Write the collected summary lines to a .txt file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [✓] Summary report saved → {OUTPUT_PATH}")


def main():
    # ── Load Dataset ─────────────────────────────────────────
    try:
        print(f"\nLoading dataset from: {INPUT_PATH}")
        df = pd.read_csv(INPUT_PATH)
        print(f"Dataset loaded successfully. Shape: {df.shape}\n")
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {INPUT_PATH}")
        print("  → Make sure city_day.csv is placed in datasets/raw/")
        return
    except pd.errors.EmptyDataError:
        print(f"\n[ERROR] The file at {INPUT_PATH} is empty.")
        return
    except pd.errors.ParserError as e:
        print(f"\n[ERROR] Could not parse CSV: {e}")
        return
    except Exception as e:
        print(f"\n[ERROR] Unexpected error while loading: {e}")
        return

    # ── Build and Print Summary ──────────────────────────────
    try:
        lines = build_summary(df)
    except Exception as e:
        print(f"\n[ERROR] Failed during summary generation: {e}")
        return

    # ── Save Report to File ──────────────────────────────────
    try:
        save_report(lines)
    except PermissionError:
        print(f"\n[ERROR] Permission denied writing to {OUTPUT_PATH}")
    except Exception as e:
        print(f"\n[ERROR] Could not save report: {e}")


if __name__ == "__main__":
    main()