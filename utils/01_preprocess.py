"""
AirSense AI – Urban Air Quality Intelligence Platform
=====================================================
Script  : utils/01_preprocess.py
Purpose : Load, validate, clean, and feature-engineer city_day.csv
          according to PREPROCESSING_SPECIFICATION.md.
          Saves cleaned CSV and a preprocessing report.

Author  : AirSense AI Engineering Team
"""

import logging
import os
import time
from datetime import datetime
from typing import Tuple

import pandas as pd

# ── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Approved file paths (only hardcoded values permitted) ────────────────────
INPUT_PATH  = os.path.join("datasets", "raw", "city_day.csv")
OUTPUT_PATH = os.path.join("datasets", "processed", "clean_city_day.csv")
REPORT_PATH = os.path.join("datasets", "reports", "preprocessing_report.txt")

# ── Column specification ─────────────────────────────────────────────────────
REQUIRED_COLUMNS: list[str] = [
    "City", "Date", "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
    "AQI", "AQI_Bucket",
]
COLUMN_TO_DROP: str = "Xylene"

# ── India seasonal month mapping ─────────────────────────────────────────────
# Winter      : December, January, February
# Summer      : March, April, May
# Monsoon     : June, July, August, September
# Post-Monsoon: October, November
SEASON_MAP: dict[int, str] = {
    1:  "Winter",
    2:  "Winter",
    3:  "Summer",
    4:  "Summer",
    5:  "Summer",
    6:  "Monsoon",
    7:  "Monsoon",
    8:  "Monsoon",
    9:  "Monsoon",
    10: "Post-Monsoon",
    11: "Post-Monsoon",
    12: "Winter",
}

# ── Lockdown window (inclusive) ──────────────────────────────────────────────
LOCKDOWN_START = pd.Timestamp("2020-03-01")
LOCKDOWN_END   = pd.Timestamp("2020-07-31")


# ─────────────────────────────────────────────────────────────────────────────
# 1. I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the raw CSV dataset from *path*.

    Parameters
    ----------
    path : str
        Filesystem path to city_day.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe as loaded from disk.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    pd.errors.EmptyDataError
        If the file contains no data.
    """
    logger.info("Loading dataset from: %s", path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Input file not found: '{path}'. "
            "Ensure city_day.csv is placed in datasets/raw/"
        )
    df = pd.read_csv(path)
    if df.empty:
        raise pd.errors.EmptyDataError(f"File is empty: '{path}'")
    logger.info("Dataset loaded — %d rows × %d columns", df.shape[0], df.shape[1])
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """
    Persist the cleaned dataframe to *path* as a CSV file.

    Parameters
    ----------
    df   : pd.DataFrame
        Processed dataframe to save.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Cleaned dataset saved → %s  (%d rows × %d columns)",
                path, df.shape[0], df.shape[1])


def save_report(content: str, path: str) -> None:
    """
    Write the preprocessing report text to *path*.

    Parameters
    ----------
    content : str
        Full report body as a single string.
    path    : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    logger.info("Preprocessing report saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Assert that every column in *required* is present in *df*.

    Parameters
    ----------
    df       : pd.DataFrame
        Dataframe to inspect.
    required : list[str]
        Column names that must be present.

    Raises
    ------
    ValueError
        If one or more required columns are absent.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Required column(s) missing from dataset: {missing}\n"
            f"Columns present: {list(df.columns)}"
        )
    logger.info("Column validation passed — all %d required columns present.",
                len(required))


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEANING STEPS
# ─────────────────────────────────────────────────────────────────────────────

def remove_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows (all column values identical).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with duplicate rows removed.
    int
        Number of rows that were removed.
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates()
    removed  = initial_count - len(df_clean)
    if removed:
        logger.info("Duplicate rows removed: %d", removed)
    else:
        logger.info("No exact duplicate rows found.")
    return df_clean, removed


def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Drop *column* from *df* if it exists.

    Parameters
    ----------
    df     : pd.DataFrame
        Input dataframe.
    column : str
        Name of the column to remove.

    Returns
    -------
    pd.DataFrame
        Dataframe without *column*.
    """
    if column in df.columns:
        df = df.drop(columns=[column])
        logger.info("Column dropped: '%s'", column)
    else:
        logger.warning("Column '%s' not found — skipping drop.", column)
    return df


def parse_dates(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """
    Convert *date_column* to ``datetime64`` in-place.

    Rows where the date cannot be parsed are logged as warnings;
    no rows are silently dropped — downstream steps handle NaT.

    Parameters
    ----------
    df          : pd.DataFrame
        Input dataframe.
    date_column : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Dataframe with *date_column* cast to datetime.
    """
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    nat_count = df[date_column].isna().sum()
    if nat_count:
        logger.warning(
            "%d row(s) have an unparseable date and will contain NaT in '%s'.",
            nat_count, date_column,
        )
    else:
        logger.info("Date column parsed successfully — 0 NaT values.")
    return df


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort *df* by City (ascending) then Date (ascending).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Sorted dataframe with reset integer index.
    """
    df = df.sort_values(by=["City", "Date"], ascending=[True, True])
    df = df.reset_index(drop=True)
    logger.info("Data sorted by City → Date.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_calendar_features(df: pd.DataFrame,
                                date_column: str = "Date") -> pd.DataFrame:
    """
    Derive calendar-based features from *date_column*.

    New columns added
    -----------------
    Year       : int   — calendar year
    Month      : int   — calendar month (1–12)
    Day        : int   — day of month (1–31)
    DayOfWeek  : int   — Monday=0 … Sunday=6
    IsWeekend  : int   — 1 if Saturday or Sunday, else 0

    Parameters
    ----------
    df          : pd.DataFrame
    date_column : str

    Returns
    -------
    pd.DataFrame
    """
    dt = df[date_column].dt
    df["Year"]      = dt.year
    df["Month"]     = dt.month
    df["Day"]       = dt.day
    df["DayOfWeek"] = dt.dayofweek          # Monday=0, Sunday=6
    df["IsWeekend"] = (dt.dayofweek >= 5).astype(int)
    logger.info("Calendar features created: Year, Month, Day, DayOfWeek, IsWeekend.")
    return df


def engineer_season(df: pd.DataFrame, season_map: dict[int, str]) -> pd.DataFrame:
    """
    Assign an India-appropriate season label based on the Month column.

    Season mapping
    --------------
    Winter       : Dec, Jan, Feb
    Summer       : Mar, Apr, May
    Monsoon      : Jun, Jul, Aug, Sep
    Post-Monsoon : Oct, Nov

    Parameters
    ----------
    df         : pd.DataFrame  — must contain a 'Month' column.
    season_map : dict[int, str]

    Returns
    -------
    pd.DataFrame
        Dataframe with new 'Season' column (str).
    """
    df["Season"] = df["Month"].map(season_map)
    unmapped = df["Season"].isna().sum()
    if unmapped:
        logger.warning("%d row(s) could not be mapped to a season.", unmapped)
    else:
        logger.info("Season column created using India seasonal calendar.")
    return df


def engineer_lockdown_flag(
    df: pd.DataFrame,
    date_column: str = "Date",
    start: pd.Timestamp = LOCKDOWN_START,
    end: pd.Timestamp   = LOCKDOWN_END,
) -> pd.DataFrame:
    """
    Create a binary ``is_lockdown`` column.

    Logic
    -----
    is_lockdown = 1  if Date falls within [LOCKDOWN_START, LOCKDOWN_END]
                      (March 2020 – July 2020, both months inclusive)
    is_lockdown = 0  otherwise

    Parameters
    ----------
    df          : pd.DataFrame
    date_column : str
    start       : pd.Timestamp — first day of lockdown window
    end         : pd.Timestamp — last day of lockdown window

    Returns
    -------
    pd.DataFrame
    """
    df["is_lockdown"] = (
        (df[date_column] >= start) & (df[date_column] <= end)
    ).astype(int)

    lockdown_rows = df["is_lockdown"].sum()
    logger.info(
        "is_lockdown column created — %d rows flagged as lockdown period "
        "(%s → %s).",
        lockdown_rows,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    initial_rows:     int,
    final_rows:       int,
    duplicates_removed: int,
    columns_dropped:  list[str],
    columns_created:  list[str],
    elapsed_seconds:  float,
) -> str:
    """
    Compose the preprocessing report as a formatted string.

    Parameters
    ----------
    initial_rows       : int
    final_rows         : int
    duplicates_removed : int
    columns_dropped    : list[str]
    columns_created    : list[str]
    elapsed_seconds    : float

    Returns
    -------
    str
        Multi-line report body.
    """
    sep  = "=" * 62
    sep2 = "-" * 62

    lines = [
        sep,
        "  AIRSENSE AI – PREPROCESSING REPORT",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Script    : utils/01_preprocess.py",
        sep,
        "",
        "[1] DATASET DIMENSIONS",
        sep2,
        f"  Initial row count       : {initial_rows:>10,}",
        f"  Final row count         : {final_rows:>10,}",
        f"  Net rows removed        : {initial_rows - final_rows:>10,}",
        "",
        "[2] DUPLICATE ROWS",
        sep2,
        f"  Exact duplicates removed: {duplicates_removed:>10,}",
        "",
        "[3] COLUMNS REMOVED",
        sep2,
    ]

    if columns_dropped:
        for col in columns_dropped:
            lines.append(f"  ✗  {col}")
    else:
        lines.append("  (none)")

    lines += [
        "",
        "[4] COLUMNS CREATED",
        sep2,
    ]

    if columns_created:
        for col in columns_created:
            lines.append(f"  ✓  {col}")
    else:
        lines.append("  (none)")

    lines += [
        "",
        "[5] PROCESSING TIME",
        sep2,
        f"  Elapsed time            : {elapsed_seconds:.4f} seconds",
        "",
        "[6] OUTPUT FILES",
        sep2,
        f"  Cleaned CSV  : {OUTPUT_PATH}",
        f"  This report  : {REPORT_PATH}",
        "",
        sep,
        "  Preprocessing complete — no imputation, scaling, or encoding applied.",
        sep,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Execute the full preprocessing pipeline end-to-end.

    Steps
    -----
    1.  Load raw CSV
    2.  Validate required columns
    3.  Record initial row count
    4.  Remove exact duplicate rows
    5.  Drop the Xylene column
    6.  Parse Date column to datetime
    7.  Sort by City → Date
    8.  Engineer calendar features (Year, Month, Day, DayOfWeek, IsWeekend)
    9.  Engineer Season column
    10. Engineer is_lockdown flag
    11. Save cleaned CSV
    12. Build and save preprocessing report
    """
    logger.info("=" * 62)
    logger.info("  AirSense AI — Preprocessing Pipeline START")
    logger.info("=" * 62)

    start_time = time.perf_counter()

    # ── Step 1: Load ─────────────────────────────────────────
    df = load_dataset(INPUT_PATH)

    # ── Step 2: Validate ─────────────────────────────────────
    validate_columns(df, REQUIRED_COLUMNS)

    # ── Step 3: Record initial row count ─────────────────────
    initial_rows = len(df)

    # ── Step 4: Remove exact duplicates ──────────────────────
    df, duplicates_removed = remove_exact_duplicates(df)

    # ── Step 5: Drop Xylene column ───────────────────────────
    columns_dropped: list[str] = [COLUMN_TO_DROP]
    df = drop_column(df, COLUMN_TO_DROP)

    # ── Step 6: Parse Date ───────────────────────────────────
    df = parse_dates(df, date_column="Date")

    # ── Step 7: Sort ─────────────────────────────────────────
    df = sort_data(df)

    # ── Steps 8–10: Feature engineering ─────────────────────
    columns_created: list[str] = [
        "Year", "Month", "Day", "DayOfWeek", "IsWeekend",
        "Season", "is_lockdown",
    ]

    df = engineer_calendar_features(df, date_column="Date")
    df = engineer_season(df, season_map=SEASON_MAP)
    df = engineer_lockdown_flag(df, date_column="Date")

    # ── Step 11: Save cleaned CSV ────────────────────────────
    save_dataset(df, OUTPUT_PATH)

    # ── Step 12: Report ──────────────────────────────────────
    elapsed = time.perf_counter() - start_time

    report_text = build_report(
        initial_rows=initial_rows,
        final_rows=len(df),
        duplicates_removed=duplicates_removed,
        columns_dropped=columns_dropped,
        columns_created=columns_created,
        elapsed_seconds=elapsed,
    )

    # Echo report to console
    print()
    print(report_text)
    print()

    # Persist report
    save_report(report_text, REPORT_PATH)

    logger.info("=" * 62)
    logger.info("  AirSense AI — Preprocessing Pipeline COMPLETE (%.4fs)", elapsed)
    logger.info("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point — wraps run_pipeline() with top-level error handling."""
    try:
        run_pipeline()
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        raise SystemExit(1) from exc
    except pd.errors.EmptyDataError as exc:
        logger.error("Dataset is empty: %s", exc)
        raise SystemExit(1) from exc
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        raise SystemExit(1) from exc
    except PermissionError as exc:
        logger.error("Permission denied when writing output: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error during preprocessing: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()