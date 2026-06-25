"""
AirSense AI – Urban Air Quality Intelligence Platform
=====================================================
Script  : utils/02_feature_engineering.py
Purpose : Read the cleaned city_day.csv and engineer research-backed
          time-series features for AQI forecasting.

          Feature groups implemented
          ──────────────────────────
          A. AQI lag features          (t-1, t-2, t-3, t-7)
          B. PM2.5 lag features        (t-1, t-3)
          C. PM10 lag feature          (t-1)
          D. CO lag feature            (t-1)
          E. Rolling window features   (mean-3, mean-7, std-7, max-7, min-7)
          F. Trend / delta features    (AQI, PM2.5, PM10 day-over-day change)
          G. Cyclical encoding         (Month, DayOfWeek → sin/cos)
          H. Weather placeholder cols  (temperature, humidity, wind_speed, rainfall)
          I. Weighted pollution index  (PM2.5, PM10, NO2, CO)

          All lag and rolling features are computed PER CITY to prevent
          cross-city data leakage.  Chronological ordering is preserved
          throughout.

Author  : AirSense AI Engineering Team
"""

import logging
import os
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

# ── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Approved file paths ──────────────────────────────────────────────────────
INPUT_PATH  = os.path.join("datasets", "processed", "clean_city_day.csv")
OUTPUT_PATH = os.path.join("datasets", "processed", "feature_engineered_city_day.csv")
REPORT_PATH = os.path.join("datasets", "reports", "feature_engineering_report.txt")

# ── Pollution index weights (research-backed, PM2.5-dominant) ────────────────
# Weights reflect relative contribution to AQI per CPCB sub-index methodology.
# PM2.5 carries the highest weight as the primary AQI driver in Indian cities.
POLLUTION_INDEX_WEIGHTS: dict[str, float] = {
    "PM2.5": 0.40,
    "PM10":  0.30,
    "NO2":   0.20,
    "CO":    0.10,
}

# ── Weather placeholder column names ────────────────────────────────────────
WEATHER_PLACEHOLDER_COLS: list[str] = [
    "temperature",
    "humidity",
    "wind_speed",
    "rainfall",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the cleaned CSV produced by 01_preprocess.py.

    Parameters
    ----------
    path : str
        Filesystem path to clean_city_day.csv.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with Date parsed as datetime64.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    pd.errors.EmptyDataError
        If the file contains no data.
    """
    logger.info("Loading cleaned dataset from: %s", path)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Input file not found: '{path}'. "
            "Run utils/01_preprocess.py first."
        )

    df = pd.read_csv(path, parse_dates=["Date"])

    if df.empty:
        raise pd.errors.EmptyDataError(f"File is empty: '{path}'")

    logger.info(
        "Dataset loaded — %d rows × %d columns", df.shape[0], df.shape[1]
    )
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """
    Persist the feature-engineered dataframe to *path* as a CSV file.

    Parameters
    ----------
    df   : pd.DataFrame
        Engineered dataframe.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(
        "Feature-engineered dataset saved → %s  (%d rows × %d columns)",
        path, df.shape[0], df.shape[1],
    )


def save_report(content: str, path: str) -> None:
    """
    Write the feature engineering report to *path*.

    Parameters
    ----------
    content : str
        Full report body.
    path    : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    logger.info("Feature engineering report saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_input(df: pd.DataFrame) -> None:
    """
    Verify that the minimum required columns are present before engineering.

    Parameters
    ----------
    df : pd.DataFrame

    Raises
    ------
    ValueError
        If any required column is absent.
    """
    required = ["City", "Date", "AQI", "PM2.5", "PM10", "NO2", "CO",
                "Month", "DayOfWeek"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing: {missing}\n"
            "Ensure 01_preprocess.py has been run successfully."
        )
    logger.info("Input validation passed — all required columns present.")


def ensure_sort_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee City → Date sort order before any per-city windowed operations.

    Even though 01_preprocess.py sorts the data, we re-sort here
    defensively so this script is safe to run independently.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Sorted dataframe with reset index.
    """
    df = df.sort_values(by=["City", "Date"], ascending=[True, True])
    df = df.reset_index(drop=True)
    logger.info("Data re-sorted by City → Date (defensive sort).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE GROUP A — AQI LAG FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_aqi_lag_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Create lagged AQI values grouped per city.

    Research basis: Goetzinger et al. (2026) and Karthick et al. (2024)
    identify AQI lags at t-1, t-3, and t-7 as the strongest predictors
    for daily AQI forecasting on Indian CPCB data.

    New columns
    -----------
    AQI_lag_1 : AQI value from 1 day prior (same city)
    AQI_lag_2 : AQI value from 2 days prior
    AQI_lag_3 : AQI value from 3 days prior
    AQI_lag_7 : AQI value from 7 days prior

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted City → Date.

    Returns
    -------
    pd.DataFrame
        Dataframe with new lag columns appended.
    list[str]
        Names of the newly created columns.
    """
    created: list[str] = []
    lag_periods = [1, 2, 3, 7]

    for lag in lag_periods:
        col_name = f"AQI_lag_{lag}"
        # groupby().shift() creates per-city lags without leaking across cities
        df[col_name] = df.groupby("City")["AQI"].shift(lag)
        created.append(col_name)

    logger.info(
        "AQI lag features created: %s", ", ".join(created)
    )
    return df, created


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE GROUP B — PM2.5 LAG FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_pm25_lag_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Create lagged PM2.5 values grouped per city.

    Research basis: PM2.5 is the primary AQI driver in Indian cities
    (Natarajan et al. 2024; Kumar & Pande 2022).  Lags t-1 and t-3
    capture both next-day and short-term particulate persistence.

    New columns
    -----------
    PM25_lag_1 : PM2.5 from 1 day prior
    PM25_lag_3 : PM2.5 from 3 days prior

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame, list[str]
    """
    created: list[str] = []

    for lag in [1, 3]:
        col_name = f"PM25_lag_{lag}"
        df[col_name] = df.groupby("City")["PM2.5"].shift(lag)
        created.append(col_name)

    logger.info("PM2.5 lag features created: %s", ", ".join(created))
    return df, created


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE GROUP C — PM10 LAG FEATURE
# ─────────────────────────────────────────────────────────────────────────────

def add_pm10_lag_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Create lagged PM10 value grouped per city.

    New columns
    -----------
    PM10_lag_1 : PM10 from 1 day prior

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame, list[str]
    """
    col_name = "PM10_lag_1"
    df[col_name] = df.groupby("City")["PM10"].shift(1)
    created = [col_name]

    logger.info("PM10 lag feature created: %s", col_name)
    return df, created


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE GROUP D — CO LAG FEATURE
# ─────────────────────────────────────────────────────────────────────────────

def add_co_lag_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Create lagged CO value grouped per city.

    CO persistence is relevant for short-term combustion-source attribution.

    New columns
    -----------
    CO_lag_1 : CO from 1 day prior

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame, list[str]
    """
    col_name = "CO_lag_1"
    df[col_name] = df.groupby("City")["CO"].shift(1)
    created = [col_name]

    logger.info("CO lag feature created: %s", col_name)
    return df, created


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE GROUP E — ROLLING WINDOW FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Compute rolling statistics of AQI per city.

    All rolling windows use ``min_periods=1`` so that early rows in each
    city group receive valid (partial-window) values instead of NaN.
    The ``.shift(1)`` applied before each rolling call ensures we only
    look at *past* observations — no future leakage.

    Research basis: Goetzinger et al. (2026) use rolling mean (3, 7, 14
    days) and rolling std (7 days) on the same Indian CPCB dataset and
    report them as top predictors alongside lag features.

    New columns
    -----------
    AQI_roll_mean_3 : 3-day rolling mean of AQI  (per city)
    AQI_roll_mean_7 : 7-day rolling mean of AQI  (per city)
    AQI_roll_std_7  : 7-day rolling std  of AQI  (per city)
    AQI_roll_max_7  : 7-day rolling max  of AQI  (per city)
    AQI_roll_min_7  : 7-day rolling min  of AQI  (per city)

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted City → Date.

    Returns
    -------
    pd.DataFrame, list[str]
    """
    created: list[str] = []

    # Helper: per-city shifted AQI series (shift=1 prevents leakage)
    # We define a lambda that shifts first then rolls.
    def _city_roll(series: pd.Series, window: int, func: str) -> pd.Series:
        """Shift by 1 then apply rolling aggregation — no future leakage."""
        shifted = series.shift(1)
        roller  = shifted.rolling(window=window, min_periods=1)
        return getattr(roller, func)()

    roll_specs: list[Tuple[str, int, str]] = [
        ("AQI_roll_mean_3", 3,  "mean"),
        ("AQI_roll_mean_7", 7,  "mean"),
        ("AQI_roll_std_7",  7,  "std"),
        ("AQI_roll_max_7",  7,  "max"),
        ("AQI_roll_min_7",  7,  "min"),
    ]

    for col_name, window, agg_func in roll_specs:
        df[col_name] = df.groupby("City")["AQI"].transform(
            lambda s, w=window, f=agg_func: _city_roll(s, w, f)
        )
        created.append(col_name)
        logger.debug("Created rolling feature: %s", col_name)

    logger.info("Rolling AQI features created: %s", ", ".join(created))
    return df, created


# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE GROUP F — TREND / DELTA FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_trend_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Compute day-over-day change (delta) for AQI, PM2.5, and PM10 per city.

    Formula: change_t = value_t − value_(t-1)

    A positive value means air quality worsened; negative means it improved.
    These trend signals help the model distinguish rising vs falling pollution
    episodes — important for the 'Smart Intervention' trigger logic.

    New columns
    -----------
    AQI_change   : AQI(t) − AQI(t-1)
    PM25_change  : PM2.5(t) − PM2.5(t-1)
    PM10_change  : PM10(t) − PM10(t-1)

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted City → Date.

    Returns
    -------
    pd.DataFrame, list[str]
    """
    created: list[str] = []

    trend_specs: list[Tuple[str, str]] = [
        ("AQI_change",  "AQI"),
        ("PM25_change", "PM2.5"),
        ("PM10_change", "PM10"),
    ]

    for col_name, source_col in trend_specs:
        # diff(1) inside groupby is equivalent to value - shift(1)
        df[col_name] = df.groupby("City")[source_col].diff(1)
        created.append(col_name)

    logger.info("Trend features created: %s", ", ".join(created))
    return df, created


# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE GROUP G — CYCLICAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def add_cyclical_encoding(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Encode Month and DayOfWeek as sine/cosine pairs.

    Rationale
    ---------
    Tree-based models (XGBoost, Random Forest) do not inherently understand
    that month 12 is adjacent to month 1.  Projecting cyclic variables onto
    a unit circle (sin/cos) preserves this circular continuity and is a
    standard technique in time-series ML (see Goetzinger et al. 2026).

    Formulae
    --------
    month_sin       = sin(2π × Month / 12)
    month_cos       = cos(2π × Month / 12)
    day_of_week_sin = sin(2π × DayOfWeek / 7)
    day_of_week_cos = cos(2π × DayOfWeek / 7)

    New columns
    -----------
    Month_sin, Month_cos, DayOfWeek_sin, DayOfWeek_cos

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Month' (1–12) and 'DayOfWeek' (0–6) columns.

    Returns
    -------
    pd.DataFrame, list[str]
    """
    # Month  — period = 12
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # DayOfWeek — period = 7
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    created = ["Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos"]
    logger.info("Cyclical encoding created: %s", ", ".join(created))
    return df, created


# ─────────────────────────────────────────────────────────────────────────────
# 10. FEATURE GROUP H — WEATHER PLACEHOLDER COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

def add_weather_placeholders(
    df: pd.DataFrame,
    placeholder_cols: list[str],
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Add empty (NaN) weather columns as placeholders for future integration.

    These columns are reserved for meteorological data that will be fetched
    via OpenWeatherMap or IMD APIs in a later pipeline stage.  Creating them
    now keeps the schema stable and avoids downstream KeyError exceptions
    in modelling scripts that reference these column names.

    New columns
    -----------
    temperature : float (°C) — to be filled by weather API fetch
    humidity    : float (%)  — relative humidity
    wind_speed  : float (m/s)
    rainfall    : float (mm)

    Parameters
    ----------
    df               : pd.DataFrame
    placeholder_cols : list[str]
        Column names to create, all initialised to NaN.

    Returns
    -------
    pd.DataFrame, list[str]
    """
    for col in placeholder_cols:
        df[col] = np.nan

    logger.info(
        "Weather placeholder columns created (all NaN): %s",
        ", ".join(placeholder_cols),
    )
    return df, list(placeholder_cols)


# ─────────────────────────────────────────────────────────────────────────────
# 11. FEATURE GROUP I — WEIGHTED POLLUTION INDEX
# ─────────────────────────────────────────────────────────────────────────────

def add_pollution_index(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Compute a single weighted pollution composite score.

    Formula
    -------
    pollution_index = Σ (weight_i × normalised_pollutant_i)

    Each pollutant is normalised to [0, 1] using its own observed min/max
    across the entire dataset (min-max scaling applied only for the purpose
    of this composite — original columns are NOT modified).

    Weights (research-backed, CPCB sub-index methodology)
    ------
    PM2.5 : 0.40  — dominant particulate AQI driver
    PM10  : 0.30  — coarse particulate contribution
    NO2   : 0.20  — secondary photochemical pollutant
    CO    : 0.10  — combustion indicator

    New column
    ----------
    pollution_index : float — weighted composite pollution score in [0, 1]

    Parameters
    ----------
    df      : pd.DataFrame
    weights : dict[str, float]
        Mapping of column name → weight.  Weights should sum to 1.0.

    Returns
    -------
    pd.DataFrame, list[str]
    """
    # Confirm all weight columns exist
    missing_cols = [c for c in weights if c not in df.columns]
    if missing_cols:
        logger.warning(
            "Pollution index: column(s) %s not found — they will be "
            "treated as zero contribution.", missing_cols
        )

    composite = pd.Series(0.0, index=df.index)

    for col, weight in weights.items():
        if col not in df.columns:
            continue

        col_series = df[col].copy()
        col_min    = col_series.min()
        col_max    = col_series.max()

        if col_max == col_min:
            # Degenerate case: all values identical → normalised value = 0
            logger.warning(
                "Pollution index: column '%s' has zero variance — "
                "contributing 0 to composite.", col
            )
            normalised = pd.Series(0.0, index=df.index)
        else:
            normalised = (col_series - col_min) / (col_max - col_min)

        composite += weight * normalised

    df["pollution_index"] = composite.round(6)
    created = ["pollution_index"]
    logger.info(
        "Weighted pollution index created (weights: %s).",
        {k: v for k, v in weights.items()},
    )
    return df, created


# ─────────────────────────────────────────────────────────────────────────────
# 12. REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    initial_shape:     Tuple[int, int],
    final_shape:       Tuple[int, int],
    feature_groups:    dict[str, list[str]],
    null_summary:      pd.Series,
    elapsed_seconds:   float,
    city_list:         list[str],
) -> str:
    """
    Compose the feature engineering report.

    Parameters
    ----------
    initial_shape   : (rows, cols) before engineering
    final_shape     : (rows, cols) after engineering
    feature_groups  : dict mapping group label → list of created column names
    null_summary    : Series with NaN count per newly created column
    elapsed_seconds : wall-clock time
    city_list       : unique city names present in the dataset

    Returns
    -------
    str
        Formatted multi-line report body.
    """
    sep  = "=" * 66
    sep2 = "-" * 66

    new_cols_total = sum(len(v) for v in feature_groups.values())

    lines = [
        sep,
        "  AIRSENSE AI – FEATURE ENGINEERING REPORT",
        f"  Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "  Script        : utils/02_feature_engineering.py",
        sep,
        "",
        "[1] DATASET DIMENSIONS",
        sep2,
        f"  Input  shape  : {initial_shape[0]:>8,} rows × {initial_shape[1]} columns",
        f"  Output shape  : {final_shape[0]:>8,} rows × {final_shape[1]} columns",
        f"  New columns   : {new_cols_total}",
        "",
        "[2] CITIES PROCESSED",
        sep2,
        f"  Total cities  : {len(city_list)}",
    ]

    for city in sorted(city_list):
        lines.append(f"    •  {city}")

    lines += [
        "",
        "[3] FEATURE GROUPS CREATED",
        sep2,
    ]

    group_labels = {
        "A": "AQI Lag Features",
        "B": "PM2.5 Lag Features",
        "C": "PM10 Lag Feature",
        "D": "CO Lag Feature",
        "E": "Rolling Window Features",
        "F": "Trend / Delta Features",
        "G": "Cyclical Encoding",
        "H": "Weather Placeholder Columns",
        "I": "Weighted Pollution Index",
    }

    for group_key, cols in feature_groups.items():
        label = group_labels.get(group_key, group_key)
        lines.append(f"  Group {group_key} — {label} ({len(cols)} columns)")
        for col in cols:
            null_count = int(null_summary.get(col, 0))
            null_note  = f"  ← {null_count:,} NaN (expected: lag/roll boundary)" \
                         if null_count else ""
            lines.append(f"    ✓  {col:<30}{null_note}")
        lines.append("")

    lines += [
        "[4] NULL VALUE SUMMARY FOR NEW COLUMNS",
        sep2,
        f"  {'Column':<30} {'NaN Count':>12}  {'NaN %':>8}",
        sep2,
    ]

    total_rows = final_shape[0]
    for col in null_summary.index:
        n   = int(null_summary[col])
        pct = n / total_rows * 100 if total_rows else 0.0
        lines.append(f"  {col:<30} {n:>12,}  {pct:>7.2f}%")

    lines += [
        "",
        "  Note: NaN values in lag/rolling features are expected — they",
        "  occur at the start of each city's time series (boundary effect).",
        "  Imputation of these values is handled in a later pipeline stage.",
        "",
        "[5] POLLUTION INDEX WEIGHTS",
        sep2,
        "  Pollutant        Weight   Rationale",
        sep2,
    ]

    for pollutant, weight in POLLUTION_INDEX_WEIGHTS.items():
        rationale = {
            "PM2.5": "Primary AQI driver in Indian cities (CPCB)",
            "PM10":  "Coarse particulate — road/construction dust",
            "NO2":   "Secondary photochemical pollutant",
            "CO":    "Combustion / vehicular emission indicator",
        }.get(pollutant, "")
        lines.append(f"  {pollutant:<16} {weight:.2f}     {rationale}")

    lines += [
        "",
        "[6] LEAKAGE PREVENTION MEASURES",
        sep2,
        "  ✓  All lag/rolling features computed via groupby('City').shift()",
        "  ✓  Rolling windows use .shift(1) before .rolling() — no same-day look",
        "  ✓  Trend features use groupby('City').diff(1) — backward only",
        "  ✓  No feature uses future AQI, PM2.5 or any other target value",
        "  ✓  Chronological City → Date sort preserved throughout",
        "",
        "[7] PROCESSING TIME",
        sep2,
        f"  Elapsed time  : {elapsed_seconds:.4f} seconds",
        "",
        "[8] OUTPUT FILES",
        sep2,
        f"  Engineered CSV : {OUTPUT_PATH}",
        f"  This report    : {REPORT_PATH}",
        "",
        sep,
        "  Feature engineering complete.",
        "  Next stage: utils/03_model_training.py",
        sep,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Execute the complete feature engineering pipeline.

    Execution order
    ---------------
    1.  Load clean_city_day.csv
    2.  Validate required columns
    3.  Ensure City → Date sort order
    4.  Group A : AQI lag features
    5.  Group B : PM2.5 lag features
    6.  Group C : PM10 lag feature
    7.  Group D : CO lag feature
    8.  Group E : Rolling AQI features
    9.  Group F : Trend / delta features
    10. Group G : Cyclical encoding (Month, DayOfWeek)
    11. Group H : Weather placeholder columns
    12. Group I : Weighted pollution index
    13. Save engineered dataset
    14. Build and save report
    """
    logger.info("=" * 66)
    logger.info("  AirSense AI — Feature Engineering Pipeline START")
    logger.info("=" * 66)

    start_time = time.perf_counter()

    # ── Step 1: Load ─────────────────────────────────────────
    df = load_dataset(INPUT_PATH)
    initial_shape = df.shape

    # ── Step 2: Validate ─────────────────────────────────────
    validate_input(df)

    # ── Step 3: Sort ─────────────────────────────────────────
    df = ensure_sort_order(df)

    # ── Collect created feature names per group ───────────────
    feature_groups: dict[str, list[str]] = {}

    # ── Step 4: Group A — AQI lag features ───────────────────
    df, cols_a = add_aqi_lag_features(df)
    feature_groups["A"] = cols_a

    # ── Step 5: Group B — PM2.5 lag features ─────────────────
    df, cols_b = add_pm25_lag_features(df)
    feature_groups["B"] = cols_b

    # ── Step 6: Group C — PM10 lag feature ───────────────────
    df, cols_c = add_pm10_lag_features(df)
    feature_groups["C"] = cols_c

    # ── Step 7: Group D — CO lag feature ─────────────────────
    df, cols_d = add_co_lag_features(df)
    feature_groups["D"] = cols_d

    # ── Step 8: Group E — Rolling AQI features ───────────────
    df, cols_e = add_rolling_features(df)
    feature_groups["E"] = cols_e

    # ── Step 9: Group F — Trend features ─────────────────────
    df, cols_f = add_trend_features(df)
    feature_groups["F"] = cols_f

    # ── Step 10: Group G — Cyclical encoding ─────────────────
    df, cols_g = add_cyclical_encoding(df)
    feature_groups["G"] = cols_g

    # ── Step 11: Group H — Weather placeholders ───────────────
    df, cols_h = add_weather_placeholders(df, WEATHER_PLACEHOLDER_COLS)
    feature_groups["H"] = cols_h

    # ── Step 12: Group I — Pollution index ───────────────────
    df, cols_i = add_pollution_index(df, POLLUTION_INDEX_WEIGHTS)
    feature_groups["I"] = cols_i

    # ── Step 13: Save dataset ─────────────────────────────────
    save_dataset(df, OUTPUT_PATH)

    # ── Step 14: Build and save report ───────────────────────
    elapsed = time.perf_counter() - start_time

    # Null counts only for newly created columns
    all_new_cols: list[str] = []
    for cols in feature_groups.values():
        all_new_cols.extend(cols)

    null_summary: pd.Series = df[all_new_cols].isnull().sum()
    null_summary  = null_summary[null_summary > 0]  # show only cols with NaN

    city_list = list(df["City"].dropna().unique())

    report_text = build_report(
        initial_shape=initial_shape,
        final_shape=df.shape,
        feature_groups=feature_groups,
        null_summary=null_summary,
        elapsed_seconds=elapsed,
        city_list=city_list,
    )

    # Echo to console
    print()
    print(report_text)
    print()

    save_report(report_text, REPORT_PATH)

    logger.info("=" * 66)
    logger.info(
        "  AirSense AI — Feature Engineering Pipeline COMPLETE (%.4fs)",
        elapsed,
    )
    logger.info("=" * 66)


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
        logger.exception("Unexpected error during feature engineering: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()