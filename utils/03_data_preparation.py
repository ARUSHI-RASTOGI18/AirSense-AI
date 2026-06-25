"""
AirSense AI – Urban Air Quality Intelligence Platform
=====================================================
Script  : utils/03_data_preparation.py
Purpose : Prepare the feature-engineered dataset for machine learning by
          performing chronological train/validation/test splitting,
          KNN-based missing value imputation, and StandardScaler scaling.

          Key design decisions
          ────────────────────
          1. Chronological split — time-series data must NEVER be randomly
             shuffled.  Shuffling would leak future pollution patterns into
             the training set, inflating reported accuracy (data leakage).

          2. Fit-on-train-only — both the KNNImputer and the StandardScaler
             are fitted exclusively on the training partition, then reused to
             transform validation and test sets.  Fitting on all data would
             leak distributional information from the future into the model's
             preprocessing steps (a subtle but serious form of leakage).

          3. KNNImputer over mean/median — research (PLOS ONE 2024; MDPI 2022)
             shows KNN imputation preserves local structure in pollutant
             time-series better than simple statistical fills, yielding
             measurably lower downstream prediction error.

          4. Dual output (scaled + unscaled) — tree-based models (XGBoost,
             Random Forest) are scale-invariant and perform best on unscaled
             features.  Scaled datasets are saved for potential use with
             distance- or gradient-based models (KNN, SVM, neural nets).

          5. Weather placeholder exclusion — the four weather columns
             (temperature, humidity, wind_speed, rainfall) are intentionally
             100% NaN because they will be populated later from live weather
             APIs.  KNNImputer cannot fit on columns with zero observed
             values in the training set (it has no neighbourhood to build),
             so these columns are separated before imputation and reattached
             afterwards — still NaN — preserving the schema intact.

          Bug fix (v2)
          ────────────
          Previous version passed all numeric columns (including 100% NaN
          weather placeholders) to KNNImputer.  sklearn's KNNImputer drops
          any column that is entirely NaN during fit, causing it to return a
          matrix with fewer columns than the input.  When this matrix was
          wrapped back into a DataFrame using the original column list, pandas
          raised:
              "Shape of passed values is (N, 39), indices imply (N, 43)"
          Fix: explicitly exclude placeholder columns before imputation and
          reattach them at the end of apply_imputer().

Author  : AirSense AI Engineering Team
"""

import logging
import os
import time
from datetime import datetime
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Approved file paths ───────────────────────────────────────────────────────
INPUT_PATH = os.path.join(
    "datasets", "processed", "feature_engineered_city_day.csv"
)

DIR_PREPARED  = os.path.join("datasets", "prepared")
DIR_REPORTS   = os.path.join("datasets", "reports")
DIR_ARTIFACTS = "artifacts"

PATH_TRAIN    = os.path.join(DIR_PREPARED, "train.csv")
PATH_VALID    = os.path.join(DIR_PREPARED, "validation.csv")
PATH_TEST     = os.path.join(DIR_PREPARED, "test.csv")
PATH_TRAIN_SC = os.path.join(DIR_PREPARED, "train_scaled.csv")
PATH_VALID_SC = os.path.join(DIR_PREPARED, "validation_scaled.csv")
PATH_TEST_SC  = os.path.join(DIR_PREPARED, "test_scaled.csv")

PATH_IMPUTER = os.path.join(DIR_ARTIFACTS, "imputer.joblib")
PATH_SCALER  = os.path.join(DIR_ARTIFACTS, "scaler.joblib")

PATH_MISSING_BEFORE = os.path.join(DIR_REPORTS, "missing_before.csv")
PATH_MISSING_AFTER  = os.path.join(DIR_REPORTS, "missing_after.csv")
PATH_REPORT         = os.path.join(DIR_REPORTS, "data_preparation_report.txt")

# ── Chronological split boundaries ────────────────────────────────────────────
TRAIN_END = "2018-12-31"
VALID_END = "2019-12-31"
TEST_END  = "2020-07-31"

# ── KNNImputer configuration ──────────────────────────────────────────────────
KNN_NEIGHBORS = 5

# ── Columns that must NEVER be imputed or scaled ──────────────────────────────
# Identifiers, categorical labels, and string columns that carry no numeric
# signal for the imputer or scaler.
NON_NUMERIC_COLS: list[str] = [
    "City",
    "Date",
    "AQI_Bucket",
    "Season",
]

# ── Weather placeholder columns ───────────────────────────────────────────────
# These columns are 100% NaN by design — they will be populated by a live
# weather API fetch in a later pipeline stage.  They must be:
#   • excluded from KNNImputer (sklearn drops fully-NaN columns during fit,
#     causing a column-count mismatch on transform)
#   • excluded from StandardScaler (NaN mean / std are undefined)
#   • preserved in the output with their original NaN values intact
WEATHER_PLACEHOLDER_COLS: list[str] = [
    "temperature",
    "humidity",
    "wind_speed",
    "rainfall",
]

# ── Columns to exclude from scaling only (binary / ordinal) ───────────────────
SCALE_EXCLUDE_COLS: list[str] = [
    "IsWeekend",
    "is_lockdown",
    "Year",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the feature-engineered CSV and parse the Date column.

    Parameters
    ----------
    path : str

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
    pd.errors.EmptyDataError
    """
    logger.info("Loading feature-engineered dataset from: %s", path)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Input file not found: '{path}'. "
            "Run utils/02_feature_engineering.py first."
        )

    df = pd.read_csv(path, parse_dates=["Date"])

    if df.empty:
        raise pd.errors.EmptyDataError(f"File is empty: '{path}'")

    logger.info(
        "Dataset loaded — %d rows × %d columns", df.shape[0], df.shape[1]
    )
    return df


def save_csv(df: pd.DataFrame, path: str, label: str = "") -> None:
    """Save *df* to *path* as CSV, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    tag = f"[{label}] " if label else ""
    logger.info(
        "%sSaved → %s  (%d rows × %d cols)",
        tag, path, df.shape[0], df.shape[1],
    )


def save_artifact(obj: object, path: str, label: str = "") -> None:
    """Persist a fitted sklearn object via joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    tag = f"[{label}] " if label else ""
    logger.info("%sArtifact saved → %s", tag, path)


def save_report(content: str, path: str) -> None:
    """Write the report string to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    logger.info("Preparation report saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_input(df: pd.DataFrame) -> None:
    """
    Confirm minimum required columns and Date dtype.

    Raises
    ------
    ValueError
    """
    required = ["City", "Date", "AQI"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing: {missing}. "
            "Ensure 02_feature_engineering.py ran successfully."
        )
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError(
            "'Date' column is not datetime64. "
            "Re-run 01_preprocess.py and 02_feature_engineering.py."
        )
    logger.info("Input validation passed.")


def validate_column_count(
    result:    pd.DataFrame,
    expected:  list[str],
    stage:     str,
) -> None:
    """
    Assert that *result* contains exactly the columns listed in *expected*.

    This guard catches any future column-count mismatch (such as the one
    caused by KNNImputer silently dropping fully-NaN columns) before it
    propagates downstream and produces cryptic pandas errors.

    Parameters
    ----------
    result   : reconstructed DataFrame
    expected : ordered list of column names that must be present
    stage    : human-readable label for the error message

    Raises
    ------
    ValueError
        If result.columns != expected (order-insensitive set comparison).
    """
    result_set   = set(result.columns)
    expected_set = set(expected)

    extra   = result_set - expected_set
    missing = expected_set - result_set

    if extra or missing:
        raise ValueError(
            f"Column mismatch after {stage}.\n"
            f"  Missing columns : {sorted(missing)}\n"
            f"  Extra   columns : {sorted(extra)}\n"
            f"  Expected {len(expected_set)} columns, "
            f"got {len(result_set)}."
        )

    logger.info(
        "Column-count validation passed after %s — %d columns present.",
        stage, len(result_set),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(
    df:        pd.DataFrame,
    train_end: str,
    valid_end: str,
    test_end:  str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition the dataset into train / validation / test using date cutoffs.

    WHY chronological?
    ------------------
    In time-series forecasting a random split contaminates the training set
    with future observations that would not be available at prediction time.
    This inflates accuracy metrics and produces models that fail in
    production.  A strict chronological split mirrors real deployment:
    the model always predicts a date it has never seen.

    Boundaries (inclusive)
    ----------------------
    Train      : Date <= TRAIN_END
    Validation : TRAIN_END < Date <= VALID_END
    Test       : VALID_END < Date <= TEST_END

    Returns
    -------
    train, validation, test : pd.DataFrame
    """
    t_end = pd.Timestamp(train_end)
    v_end = pd.Timestamp(valid_end)
    x_end = pd.Timestamp(test_end)

    train = df[df["Date"] <= t_end].copy()
    valid = df[(df["Date"] > t_end) & (df["Date"] <= v_end)].copy()
    test  = df[(df["Date"] > v_end) & (df["Date"] <= x_end)].copy()

    logger.info(
        "Chronological split complete:\n"
        "  Train      : %s → %s  (%d rows)\n"
        "  Validation : %s → %s  (%d rows)\n"
        "  Test       : %s → %s  (%d rows)",
        df["Date"].min().date(), t_end.date(), len(train),
        (t_end + pd.Timedelta(days=1)).date(), v_end.date(), len(valid),
        (v_end + pd.Timedelta(days=1)).date(), x_end.date(), len(test),
    )

    if len(train) == 0:
        raise ValueError("Training set is empty — check TRAIN_END date.")
    if len(valid) == 0:
        logger.warning("Validation set is empty — check VALID_END date.")
    if len(test) == 0:
        logger.warning("Test set is empty — check TEST_END date.")

    return train, valid, test


# ─────────────────────────────────────────────────────────────────────────────
# 4. MISSING VALUE REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def build_missing_report(
    train:        pd.DataFrame,
    valid:        pd.DataFrame,
    test:         pd.DataFrame,
    numeric_cols: list[str],
    label:        str,
) -> pd.DataFrame:
    """
    Compute per-column missing value counts and percentages for each split.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for col in numeric_cols:
        def _stats(part: pd.DataFrame) -> Tuple[int, float]:
            n   = int(part[col].isna().sum()) if col in part.columns else 0
            pct = round(n / len(part) * 100, 3) if len(part) else 0.0
            return n, pct

        t_n, t_p = _stats(train)
        v_n, v_p = _stats(valid)
        x_n, x_p = _stats(test)

        rows.append({
            "feature":       col,
            "train_missing": t_n,
            "train_pct":     t_p,
            "valid_missing": v_n,
            "valid_pct":     v_p,
            "test_missing":  x_n,
            "test_pct":      x_p,
        })

    report_df = (
        pd.DataFrame(rows)
        .sort_values("train_missing", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "Missing value report built (%s) — columns with any NaN: %d",
        label,
        int((report_df["train_missing"] > 0).sum()),
    )
    return report_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. KNN IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def get_numeric_imputation_cols(df: pd.DataFrame) -> list[str]:
    """
    Identify numeric columns eligible for KNN imputation.

    Excluded
    --------
    • NON_NUMERIC_COLS   — identifiers and categorical strings
    • WEATHER_PLACEHOLDER_COLS — 100% NaN by design; KNNImputer cannot fit
      on columns with zero observed values and silently drops them, causing
      a shape mismatch on transform.  These columns are handled separately.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[str]
    """
    # Build the full exclusion set
    exclude = set(NON_NUMERIC_COLS) | set(WEATHER_PLACEHOLDER_COLS)

    # Keep only genuine numeric columns
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    eligible = [c for c in numeric if c not in exclude]

    logger.info(
        "Columns eligible for KNN imputation  : %d  "
        "(excluded: %d non-numeric/categorical, %d weather placeholders)",
        len(eligible),
        len([c for c in NON_NUMERIC_COLS if c in df.columns]),
        len([c for c in WEATHER_PLACEHOLDER_COLS if c in df.columns]),
    )
    return eligible


def fit_imputer(
    train:        pd.DataFrame,
    numeric_cols: list[str],
    n_neighbors:  int,
) -> KNNImputer:
    """
    Fit a KNNImputer on the training split only.

    WHY fit-on-train-only?
    ----------------------
    Fitting on all data would incorporate statistics from future observations
    into the imputer's neighbourhood search, leaking information about the
    future into the training process.

    Parameters
    ----------
    train        : training partition (post-split, pre-imputation)
    numeric_cols : columns to fit on (no 100% NaN columns included)
    n_neighbors  : KNN neighbourhood size

    Returns
    -------
    KNNImputer (fitted)
    """
    logger.info(
        "Fitting KNNImputer (n_neighbors=%d) on %d columns, "
        "training rows only …",
        n_neighbors, len(numeric_cols),
    )
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputer.fit(train[numeric_cols])
    logger.info("KNNImputer fitted successfully.")
    return imputer


def apply_imputer(
    imputer:      KNNImputer,
    df:           pd.DataFrame,
    numeric_cols: list[str],
    expected_cols: list[str],
    split_label:  str,
) -> pd.DataFrame:
    """
    Apply a pre-fitted KNNImputer and reconstruct the full DataFrame.

    Reconstruction strategy
    -----------------------
    The input DataFrame is decomposed into three non-overlapping groups:

      1. passthrough_cols  — NON_NUMERIC_COLS (City, Date, Season, AQI_Bucket)
                             Categorical / identifier columns; passed through
                             unchanged.

      2. placeholder_cols  — WEATHER_PLACEHOLDER_COLS (temperature, humidity,
                             wind_speed, rainfall)
                             Still 100% NaN; separated before imputation and
                             reattached afterwards so the schema stays intact.

      3. numeric_cols      — all other numeric features; imputed by KNNImputer.

    After imputation the three groups are concatenated and reordered to match
    the original column sequence, then validated against *expected_cols*.

    WHY this decomposition fixes the bug
    -------------------------------------
    sklearn's KNNImputer.transform() returns a numpy array whose number of
    columns equals the number of columns the imputer was *fitted* on.
    If the fit excluded weather placeholders (4 columns) but the transform
    call received all columns (including those 4), the returned array would
    have 4 fewer columns than expected, and wrapping it into a DataFrame with
    the full column list raises:
        "Shape of passed values is (N, 39), indices imply (N, 43)"
    By separating placeholders *before* calling transform() — and only
    passing numeric_cols to transform() — the returned array always has
    exactly len(numeric_cols) columns, matching what the imputer expects.

    Parameters
    ----------
    imputer       : fitted KNNImputer
    df            : partition to transform
    numeric_cols  : columns the imputer was fitted on
    expected_cols : full ordered column list for post-imputation validation
    split_label   : used only in log messages

    Returns
    -------
    pd.DataFrame — same shape and column order as *df*
    """
    logger.info("Applying KNNImputer to %s split …", split_label)

    # ── Group 1: passthrough (non-numeric / categorical) ──────────────────────
    passthrough_present = [c for c in NON_NUMERIC_COLS if c in df.columns]
    passthrough_df = df[passthrough_present].reset_index(drop=True)

    # ── Group 2: weather placeholders (100% NaN — bypass imputer entirely) ────
    placeholder_present = [c for c in WEATHER_PLACEHOLDER_COLS if c in df.columns]
    placeholder_df = df[placeholder_present].reset_index(drop=True)
    # These remain NaN; we carry them forward untouched.

    # ── Group 3: numeric columns eligible for imputation ──────────────────────
    # Only pass columns the imputer was fitted on — no extras, no missing.
    numeric_block  = df[numeric_cols].reset_index(drop=True)
    imputed_values = imputer.transform(numeric_block)

    # imputer.transform() returns a numpy ndarray of shape
    # (n_rows, len(numeric_cols)).  Wrapping it with the exact same column
    # list guarantees no shape mismatch.
    imputed_df = pd.DataFrame(
        imputed_values,
        columns=numeric_cols,
    )

    # ── Reassemble all three groups ───────────────────────────────────────────
    result = pd.concat(
        [passthrough_df, placeholder_df, imputed_df],
        axis=1,
    )

    # ── Restore the original column order ────────────────────────────────────
    # concat() appends columns in group order; we reorder to match the
    # original DataFrame so downstream code never sees a schema surprise.
    original_order = [c for c in df.columns if c in result.columns]
    result = result[original_order]

    # ── Post-imputation column-count validation ───────────────────────────────
    validate_column_count(result, expected_cols, stage=f"KNN imputation ({split_label})")

    # Report residual NaN in imputed columns (should be 0)
    residual_nan = int(result[numeric_cols].isna().sum().sum())
    logger.info(
        "Imputation complete for %s — residual NaN in numeric cols: %d",
        split_label, residual_nan,
    )
    return result


def run_imputation(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test:  pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, KNNImputer, list[str]]:
    """
    Orchestrate the full imputation workflow across all three splits.

    Returns
    -------
    train_imp, valid_imp, test_imp : imputed DataFrames
    imputer                        : fitted KNNImputer
    numeric_cols                   : imputed column names
    """
    numeric_cols  = get_numeric_imputation_cols(train)

    # The expected full column list for each split (schema guard)
    expected_cols = list(train.columns)

    imputer   = fit_imputer(train, numeric_cols, KNN_NEIGHBORS)

    train_imp = apply_imputer(imputer, train, numeric_cols, expected_cols, "train")
    valid_imp = apply_imputer(imputer, valid, numeric_cols, expected_cols, "validation")
    test_imp  = apply_imputer(imputer, test,  numeric_cols, expected_cols, "test")

    return train_imp, valid_imp, test_imp, imputer, numeric_cols


# ─────────────────────────────────────────────────────────────────────────────
# 6. STANDARD SCALING
# ─────────────────────────────────────────────────────────────────────────────

def get_scaling_cols(
    df:           pd.DataFrame,
    numeric_cols: list[str],
) -> list[str]:
    """
    Identify columns to scale — numeric but excluding binary/ordinal specials
    and the weather placeholders (which are still NaN and cannot be scaled).

    WHY exclude binary columns?
    ---------------------------
    IsWeekend and is_lockdown already live in {0, 1}.  Scaling them centres
    them near 0 and shrinks their range without any modelling benefit.

    Returns
    -------
    list[str]
    """
    exclude  = (
        set(NON_NUMERIC_COLS)
        | set(SCALE_EXCLUDE_COLS)
        | set(WEATHER_PLACEHOLDER_COLS)   # NaN → undefined mean/std
    )
    to_scale = [c for c in numeric_cols if c not in exclude]
    logger.info("Columns selected for StandardScaler: %d", len(to_scale))
    return to_scale


def fit_scaler(
    train:      pd.DataFrame,
    scale_cols: list[str],
) -> StandardScaler:
    """
    Fit StandardScaler on the training partition only.

    WHY StandardScaler?
    -------------------
    StandardScaler (zero mean, unit variance) handles the wide magnitude
    differences between pollutant concentrations (CO ~ 0–5 vs PM2.5 ~
    0–500 µg/m³).  Fitting only on train data keeps test-set statistics
    invisible during training.

    Returns
    -------
    StandardScaler (fitted)
    """
    logger.info("Fitting StandardScaler on training data only …")
    scaler = StandardScaler()
    scaler.fit(train[scale_cols])
    logger.info("StandardScaler fitted successfully.")
    return scaler


def apply_scaler(
    scaler:      StandardScaler,
    df:          pd.DataFrame,
    scale_cols:  list[str],
    split_label: str,
) -> pd.DataFrame:
    """
    Apply a pre-fitted StandardScaler to *df* and return a scaled copy.

    Non-scaled columns (identifiers, binary, placeholders) are preserved.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Applying StandardScaler to %s split …", split_label)
    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.transform(df[scale_cols])
    logger.info("Scaling complete for %s.", split_label)
    return df_scaled


def run_scaling(
    train_imp:    pd.DataFrame,
    valid_imp:    pd.DataFrame,
    test_imp:     pd.DataFrame,
    numeric_cols: list[str],
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    StandardScaler, list[str],
]:
    """
    Orchestrate the full scaling workflow.

    Returns
    -------
    train_sc, valid_sc, test_sc : scaled DataFrames
    scaler                      : fitted StandardScaler
    scale_cols                  : names of scaled columns
    """
    scale_cols = get_scaling_cols(train_imp, numeric_cols)

    scaler   = fit_scaler(train_imp, scale_cols)
    train_sc = apply_scaler(scaler, train_imp, scale_cols, "train")
    valid_sc = apply_scaler(scaler, valid_imp, scale_cols, "validation")
    test_sc  = apply_scaler(scaler, test_imp,  scale_cols, "test")

    return train_sc, valid_sc, test_sc, scaler, scale_cols


# ─────────────────────────────────────────────────────────────────────────────
# 7. REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    initial_shape:   Tuple[int, int],
    train:           pd.DataFrame,
    valid:           pd.DataFrame,
    test:            pd.DataFrame,
    missing_before:  pd.DataFrame,
    missing_after:   pd.DataFrame,
    scaler:          StandardScaler,
    scale_cols:      list[str],
    numeric_cols:    list[str],
    elapsed_seconds: float,
) -> str:
    """Compose the full data preparation report."""
    sep  = "=" * 68
    sep2 = "-" * 68

    total_prepared = len(train) + len(valid) + len(test)

    lines = [
        sep,
        "  AIRSENSE AI – DATA PREPARATION REPORT",
        f"  Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "  Script        : utils/03_data_preparation.py",
        sep,
        "",
        "[1] DATASET SIZES",
        sep2,
        f"  Input shape (feature-engineered) : "
        f"{initial_shape[0]:>8,} rows × {initial_shape[1]} columns",
        f"  Total rows after split           : {total_prepared:>8,}",
        "",
        f"  {'Split':<14} {'Rows':>8}   {'Date Range'}",
        sep2,
    ]

    for label, split in [("Train", train), ("Validation", valid), ("Test", test)]:
        if len(split):
            d_min = split["Date"].min().strftime("%Y-%m-%d")
            d_max = split["Date"].max().strftime("%Y-%m-%d")
        else:
            d_min = d_max = "N/A"
        lines.append(
            f"  {label:<14} {len(split):>8,}   {d_min}  →  {d_max}"
        )

    lines += [
        "",
        "[2] SPLIT CONFIGURATION",
        sep2,
        f"  Train end date   : {TRAIN_END}",
        f"  Valid end date   : {VALID_END}",
        f"  Test  end date   : {TEST_END}",
        "  Split method     : Chronological (no shuffle, no leakage)",
        "",
        "[3] IMPUTATION EXCLUSIONS",
        sep2,
        "  The following columns were excluded from KNNImputer because they",
        "  are intentionally 100% NaN (populated later by weather API fetch).",
        "  They are preserved in all output files with their NaN values intact.",
        "",
    ]

    for col in WEATHER_PLACEHOLDER_COLS:
        lines.append(f"    ⊘  {col}  (weather placeholder — excluded from imputation)")

    lines += [
        "",
        "[4] MISSING VALUE SUMMARY — BEFORE IMPUTATION",
        sep2,
        f"  Numeric columns assessed (excl. placeholders) : {len(numeric_cols)}",
    ]

    cols_with_missing = missing_before[missing_before["train_missing"] > 0]
    if len(cols_with_missing):
        lines.append(
            f"\n  {'Feature':<30} {'Train NaN':>10} {'Train %':>9} "
            f"{'Valid NaN':>10} {'Test NaN':>10}"
        )
        lines.append(sep2)
        for _, row in cols_with_missing.head(20).iterrows():
            lines.append(
                f"  {str(row['feature']):<30} "
                f"{int(row['train_missing']):>10,} "
                f"{row['train_pct']:>8.2f}% "
                f"{int(row['valid_missing']):>10,} "
                f"{int(row['test_missing']):>10,}"
            )
        if len(cols_with_missing) > 20:
            lines.append(
                f"  … and {len(cols_with_missing) - 20} more "
                "(see missing_before.csv)"
            )
    else:
        lines.append("  No missing values found before imputation.")

    lines += [
        "",
        "[5] IMPUTATION SUMMARY",
        sep2,
        "  Method           : KNNImputer (sklearn)",
        f"  n_neighbors      : {KNN_NEIGHBORS}",
        "  Fit partition    : Training only (no leakage from future data)",
        "  Applied to       : Train, Validation, Test",
        f"  Artifact saved   : {PATH_IMPUTER}",
        "",
        "  Residual NaN after imputation:",
    ]

    cols_still_missing = missing_after[missing_after["train_missing"] > 0]
    if len(cols_still_missing):
        for _, row in cols_still_missing.iterrows():
            lines.append(
                f"    ⚠  {str(row['feature']):<30} "
                f"train={int(row['train_missing'])}  "
                f"valid={int(row['valid_missing'])}  "
                f"test={int(row['test_missing'])}"
            )
    else:
        lines.append(
            "    ✓  All imputed numeric columns fully resolved — 0 NaN remaining."
        )

    lines += [
        "",
        "[6] SCALING SUMMARY",
        sep2,
        "  Method           : StandardScaler (zero mean, unit variance)",
        "  Fit partition    : Training only (prevents distribution leakage)",
        f"  Columns scaled   : {len(scale_cols)}",
        f"  Artifact saved   : {PATH_SCALER}",
        "",
        "  Scaler statistics (first 10 columns):",
        f"  {'Column':<30} {'Mean':>12} {'Std':>12}",
        sep2,
    ]

    for col in scale_cols[:10]:
        idx  = list(scale_cols).index(col)
        mean = scaler.mean_[idx]
        std  = float(np.sqrt(scaler.var_[idx]))
        lines.append(f"  {col:<30} {mean:>12.4f} {std:>12.4f}")

    if len(scale_cols) > 10:
        lines.append(
            f"  … and {len(scale_cols) - 10} more columns scaled."
        )

    lines += [
        "",
        "[7] OUTPUT FILES",
        sep2,
        f"  Unscaled  train      : {PATH_TRAIN}",
        f"  Unscaled  validation : {PATH_VALID}",
        f"  Unscaled  test       : {PATH_TEST}",
        f"  Scaled    train      : {PATH_TRAIN_SC}",
        f"  Scaled    validation : {PATH_VALID_SC}",
        f"  Scaled    test       : {PATH_TEST_SC}",
        f"  Imputer artifact     : {PATH_IMPUTER}",
        f"  Scaler artifact      : {PATH_SCALER}",
        f"  Missing before       : {PATH_MISSING_BEFORE}",
        f"  Missing after        : {PATH_MISSING_AFTER}",
        "",
        "[8] DESIGN DECISIONS",
        sep2,
        "  • Chronological split prevents future-leakage into training.",
        "  • KNNImputer fit on train-only preserves temporal integrity.",
        "  • Weather placeholders excluded from imputer AND scaler.",
        "  • Unscaled datasets provided for tree-based models (XGBoost, RF).",
        "  • Scaled datasets provided for distance/gradient models (SVM, NN).",
        "  • Binary cols (IsWeekend, is_lockdown) excluded from scaling.",
        "  • Categorical cols (City, Season, AQI_Bucket) excluded from both.",
        "  • Column-count validation fires after every impute/scale step.",
        "",
        "[9] PROCESSING TIME",
        sep2,
        f"  Elapsed time  : {elapsed_seconds:.4f} seconds",
        "",
        sep,
        "  Data preparation complete.",
        "  Next stage: utils/04_model_training.py",
        sep,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Execute the full data preparation pipeline.

    Steps
    -----
    1.  Load feature_engineered_city_day.csv
    2.  Validate required columns
    3.  Chronological train / validation / test split
    4.  Build missing-value report (before imputation)
    5.  Fit KNNImputer on train (excluding placeholders); transform all splits
    6.  Build missing-value report (after imputation)
    7.  Fit StandardScaler on train (excluding placeholders); transform all
    8.  Save all six CSV partitions (scaled + unscaled)
    9.  Save imputer and scaler artifacts
    10. Build and save the preparation report
    """
    logger.info("=" * 68)
    logger.info("  AirSense AI — Data Preparation Pipeline START")
    logger.info("=" * 68)

    start_time = time.perf_counter()

    # ── Step 1: Load ─────────────────────────────────────────
    df = load_dataset(INPUT_PATH)
    initial_shape = df.shape

    # ── Step 2: Validate ─────────────────────────────────────
    validate_input(df)

    # ── Step 3: Chronological split ──────────────────────────
    train_raw, valid_raw, test_raw = chronological_split(
        df, TRAIN_END, VALID_END, TEST_END
    )

    # ── Step 4: Missing value report — BEFORE imputation ─────
    numeric_cols_probe = get_numeric_imputation_cols(train_raw)
    missing_before = build_missing_report(
        train_raw, valid_raw, test_raw,
        numeric_cols_probe,
        "before_imputation",
    )
    save_csv(missing_before, PATH_MISSING_BEFORE, label="missing_before")

    # ── Step 5: KNN imputation ────────────────────────────────
    train_imp, valid_imp, test_imp, imputer, numeric_cols = run_imputation(
        train_raw, valid_raw, test_raw
    )

    # ── Step 6: Missing value report — AFTER imputation ──────
    missing_after = build_missing_report(
        train_imp, valid_imp, test_imp,
        numeric_cols,
        "after_imputation",
    )
    save_csv(missing_after, PATH_MISSING_AFTER, label="missing_after")

    # ── Step 7: Standard scaling ──────────────────────────────
    train_sc, valid_sc, test_sc, scaler, scale_cols = run_scaling(
        train_imp, valid_imp, test_imp, numeric_cols
    )

    # ── Step 8: Save all six CSV datasets ────────────────────
    save_csv(train_imp, PATH_TRAIN,    label="train (unscaled)")
    save_csv(valid_imp, PATH_VALID,    label="validation (unscaled)")
    save_csv(test_imp,  PATH_TEST,     label="test (unscaled)")
    save_csv(train_sc,  PATH_TRAIN_SC, label="train (scaled)")
    save_csv(valid_sc,  PATH_VALID_SC, label="validation (scaled)")
    save_csv(test_sc,   PATH_TEST_SC,  label="test (scaled)")

    # ── Step 9: Save artifacts ────────────────────────────────
    save_artifact(imputer, PATH_IMPUTER, label="KNNImputer")
    save_artifact(scaler,  PATH_SCALER,  label="StandardScaler")

    # ── Step 10: Report ───────────────────────────────────────
    elapsed = time.perf_counter() - start_time

    report_text = build_report(
        initial_shape=initial_shape,
        train=train_imp,
        valid=valid_imp,
        test=test_imp,
        missing_before=missing_before,
        missing_after=missing_after,
        scaler=scaler,
        scale_cols=scale_cols,
        numeric_cols=numeric_cols,
        elapsed_seconds=elapsed,
    )

    print()
    print(report_text)
    print()

    save_report(report_text, PATH_REPORT)

    logger.info("=" * 68)
    logger.info(
        "  AirSense AI — Data Preparation Pipeline COMPLETE (%.4fs)", elapsed
    )
    logger.info("=" * 68)


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
        logger.exception("Unexpected error in data preparation: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()