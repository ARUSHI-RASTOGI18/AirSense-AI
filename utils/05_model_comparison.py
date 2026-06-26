"""
AirSense AI – Urban Air Quality Intelligence Platform
=====================================================
Script  : utils/05_model_comparison.py
Purpose : Enterprise ML Benchmark & Decision Engine.

          Reads all trained-model artefacts produced by 04_train_models.py,
          evaluates and ranks every model using a weighted Engineering Score,
          flags deployment risks, generates a full leaderboard, writes
          executive and detailed reports, and produces publication-quality
          comparison charts.

          This script does NOT train, retrain, or modify any model.

          Outputs produced
          ────────────────
          leaderboard/leaderboard.csv
          leaderboard/leaderboard.json
          reports/evaluation_summary.txt
          reports/evaluation_report.txt
          charts/validation_r2.png
          charts/test_rmse.png
          charts/mae.png
          charts/mape.png
          charts/training_time.png
          charts/prediction_time.png
          charts/generalization_gap.png
          charts/engineering_score.png

          Engineering Score weights
          ─────────────────────────
          Validation R²      : 35 %  — primary accuracy signal
          Test RMSE          : 25 %  — generalisation to unseen data
          Generalisation Gap : 15 %  — overfitting penalty
          Prediction Time    : 10 %  — production latency
          Training Time      :  5 %  — retraining cost
          Interpretability   : 10 %  — operational transparency

          All weights are documented in _SCORE_WEIGHTS and can be adjusted
          without touching any other part of the code.

Author  : AirSense AI Engineering Team
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on any server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Directory paths ────────────────────────────────────────────────────────────
DIR_METRICS       = "metrics"
DIR_METADATA      = "models_metadata"
DIR_PREDICTIONS   = "predictions"
DIR_LEADERBOARD   = "leaderboard"
DIR_REPORTS       = "reports"
DIR_CHARTS        = "charts"

PATH_LEADERBOARD_CSV  = os.path.join(DIR_LEADERBOARD, "leaderboard.csv")
PATH_LEADERBOARD_JSON = os.path.join(DIR_LEADERBOARD, "leaderboard.json")
PATH_EVAL_SUMMARY     = os.path.join(DIR_REPORTS, "evaluation_summary.txt")
PATH_EVAL_REPORT      = os.path.join(DIR_REPORTS, "evaluation_report.txt")

# ── Engineering Score weights ─────────────────────────────────────────────────
# Weights must sum to 1.0.  Each key maps to a normalised sub-score in [0, 1].
# Changing weights here is the ONLY place that needs to be edited.
_SCORE_WEIGHTS: Dict[str, float] = {
    "validation_r2":      0.35,   # primary accuracy on held-out 2019 data
    "test_rmse":          0.25,   # generalisation to unseen 2020 data
    "generalization_gap": 0.15,   # overfitting penalty (Train R² − Val R²)
    "prediction_time":    0.10,   # production inference latency
    "training_time":      0.05,   # retraining cost
    "interpretability":   0.10,   # transparency for urban policy use-cases
}
assert abs(sum(_SCORE_WEIGHTS.values()) - 1.0) < 1e-9, \
    "Engineering Score weights must sum to 1.0"

# ── Interpretability scores (static, domain-expert assigned) ─────────────────
# Higher = more interpretable.  Assigned by model family, not by name, so new
# model names that contain a known family keyword are matched automatically.
_INTERPRETABILITY_KEYWORDS: List[Tuple[str, float]] = [
    ("LinearRegression", 1.00),
    ("Ridge",            0.95),
    ("Lasso",            0.95),
    ("DecisionTree",     0.85),
    ("RandomForest",     0.70),
    ("ExtraTrees",       0.65),
    ("XGBoost",          0.60),
    ("LightGBM",         0.55),
    ("CatBoost",         0.55),
    ("MLP",              0.30),
]

# ── Deployment thresholds ─────────────────────────────────────────────────────
_PROD_READY_MIN_VAL_R2     = 0.85
_PROD_READY_MAX_GEN_GAP    = 0.10
_EXPERIMENTAL_MIN_VAL_R2   = 0.60

# ── Risk flag thresholds ──────────────────────────────────────────────────────
_RISK_OVERFIT_GAP      = 0.10   # Train R² − Val R² > this → overfitting risk
_RISK_SLOW_PRED_S      = 1.0    # prediction time > 1 s → slow prediction
_RISK_SLOW_TRAIN_S     = 300.0  # training time > 300 s → slow training
_RISK_WEAK_VAL_R2      = 0.70   # Val R² < 0.70 → weak performance
_RISK_HIGH_RMSE        = 50.0   # RMSE > 50 AQI units → large error

# ── Chart style ───────────────────────────────────────────────────────────────
_FIG_WIDTH  = 14
_FIG_HEIGHT = 6
_BAR_COLOR  = "#2E86AB"
_BEST_COLOR = "#27AE60"
_FONT_TITLE = 14
_FONT_AXIS  = 11


# ─────────────────────────────────────────────────────────────────────────────
# 1. I/O — LOAD METRICS & METADATA
# ─────────────────────────────────────────────────────────────────────────────

def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    """
    Safely load and validate a single JSON file.

    Returns None (with a logged warning) for any file that is missing,
    empty, or contains malformed JSON — so one corrupt file never aborts
    the entire evaluation run.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict or None
    """
    if not os.path.isfile(path):
        logger.warning("File not found: %s — skipping.", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("JSON root is not a dict in %s — skipping.", path)
            return None
        return data
    except json.JSONDecodeError as exc:
        logger.warning("Malformed JSON in %s: %s — skipping.", path, exc)
        return None
    except Exception as exc:
        logger.warning("Cannot read %s: %s — skipping.", path, exc)
        return None


def load_all_metrics(metrics_dir: str) -> List[Dict[str, Any]]:
    """
    Load all *_metrics.json files from *metrics_dir*.

    Parameters
    ----------
    metrics_dir : str

    Returns
    -------
    list[dict] — one dict per successfully loaded model
    """
    if not os.path.isdir(metrics_dir):
        raise FileNotFoundError(
            f"Metrics directory not found: '{metrics_dir}'. "
            "Run utils/04_train_models.py first."
        )

    files   = sorted(f for f in os.listdir(metrics_dir)
                     if f.endswith("_metrics.json"))
    records = []

    for fname in files:
        path   = os.path.join(metrics_dir, fname)
        record = load_json_file(path)
        if record is not None:
            records.append(record)
            logger.info("Loaded metrics: %s", fname)
        # else: already warned inside load_json_file

    logger.info("Metrics loaded: %d / %d files.", len(records), len(files))
    return records


def load_all_metadata(metadata_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all *.json files from *metadata_dir*.

    Returns
    -------
    dict[model_name, metadata_dict]
    """
    if not os.path.isdir(metadata_dir):
        logger.warning(
            "Metadata directory not found: '%s' — metadata will be omitted.",
            metadata_dir,
        )
        return {}

    files    = sorted(f for f in os.listdir(metadata_dir) if f.endswith(".json"))
    metadata = {}

    for fname in files:
        path   = os.path.join(metadata_dir, fname)
        record = load_json_file(path)
        if record is not None:
            name = record.get("model_name", fname.replace(".json", ""))
            metadata[name] = record
            logger.info("Loaded metadata: %s", fname)

    logger.info("Metadata loaded: %d entries.", len(metadata))
    return metadata


def load_predictions(
    predictions_dir: str,
    model_name:      str,
) -> Optional[pd.DataFrame]:
    """
    Load a model's prediction CSV if it exists.

    Parameters
    ----------
    predictions_dir : str
    model_name      : str

    Returns
    -------
    pd.DataFrame or None
    """
    path = os.path.join(predictions_dir, f"{model_name}_predictions.csv")
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Cannot load predictions for %s: %s", model_name, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION FROM METRICS
# ─────────────────────────────────────────────────────────────────────────────

def safe_get(record: Dict[str, Any], key: str, default: float = np.nan) -> float:
    """Return float(record[key]) or *default* if missing / non-numeric."""
    val = record.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def extract_evaluation_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten one metrics dict into a single evaluation row.

    Derived field
    -------------
    generalization_gap = train_r2 − validation_r2
    A positive gap indicates overfitting; a gap > 0.10 is flagged.

    Parameters
    ----------
    record : raw metrics dict from *_metrics.json

    Returns
    -------
    dict with consistent field names used throughout this module
    """
    train_r2 = safe_get(record, "train_r2")
    val_r2   = safe_get(record, "validation_r2")
    gen_gap  = (train_r2 - val_r2) if not (np.isnan(train_r2) or np.isnan(val_r2)) \
               else np.nan

    return {
        "model":              record.get("model_name", "Unknown"),
        "uses_scaled":        record.get("uses_scaled", False),
        "train_r2":           train_r2,
        "validation_r2":      val_r2,
        "test_r2":            safe_get(record, "test_r2"),
        "mae":                safe_get(record, "validation_mae"),
        "rmse":               safe_get(record, "validation_rmse"),
        "mape":               safe_get(record, "validation_mape"),
        "train_time_s":       safe_get(record, "train_time_s"),
        "val_pred_time_s":    safe_get(record, "validation_pred_time"),
        "generalization_gap": gen_gap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. INTERPRETABILITY SCORING
# ─────────────────────────────────────────────────────────────────────────────

def get_interpretability(model_name: str) -> float:
    """
    Return the interpretability score for *model_name* in [0, 1].

    Matching is done by checking whether the model name *contains* any
    keyword in _INTERPRETABILITY_KEYWORDS (case-sensitive, longest match
    wins).  If no keyword matches, a conservative default of 0.40 is used.

    Parameters
    ----------
    model_name : str

    Returns
    -------
    float in [0, 1]
    """
    # Sort by keyword length descending so longer/more-specific names win
    for keyword, score in sorted(
        _INTERPRETABILITY_KEYWORDS, key=lambda x: len(x[0]), reverse=True
    ):
        if keyword in model_name:
            return score
    return 0.40  # conservative default for unknown model families


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENGINEERING SCORE
# ─────────────────────────────────────────────────────────────────────────────

def normalise_col(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    Min-max normalise *series* to [0, 1].

    Parameters
    ----------
    series           : pd.Series (numeric, may contain NaN)
    higher_is_better : if True, higher raw values → higher normalised score;
                       if False (e.g. RMSE, time), lower raw values → higher score.

    Returns
    -------
    pd.Series in [0, 1] (NaN preserved)
    """
    col_min = series.min()
    col_max = series.max()

    if col_max == col_min:
        # All models are equal on this metric — assign 0.5 uniformly
        return series.apply(lambda _: 0.5 if not np.isnan(_) else np.nan)

    if higher_is_better:
        return (series - col_min) / (col_max - col_min)
    else:
        return (col_max - series) / (col_max - col_min)


def compute_engineering_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Overall Engineering Score for every model in *df*.

    Scoring methodology
    ───────────────────
    Each metric is normalised independently to [0, 1] so that metrics with
    different units (R², seconds, AQI units) become comparable.

    For metrics where HIGHER is better (R²):
        normalised = (value − min) / (max − min)

    For metrics where LOWER is better (RMSE, gap, times):
        normalised = (max − value) / (max − min)

    The interpretability score is already in [0, 1] and is not re-normalised.

    Final score = weighted sum of all normalised sub-scores.

    Parameters
    ----------
    df : DataFrame with one row per model, containing the raw metric columns

    Returns
    -------
    df with additional columns:
        norm_*             — normalised sub-scores
        interpretability   — raw interpretability score
        engineering_score  — weighted composite in [0, 1]
    """
    df = df.copy()

    # Add interpretability column
    df["interpretability"] = df["model"].apply(get_interpretability)

    # ── Normalise each component ───────────────────────────────────────────────
    # Validation R² — higher is better
    df["norm_val_r2"] = normalise_col(df["validation_r2"], higher_is_better=True)

    # Test RMSE — lower is better
    df["norm_test_rmse"] = normalise_col(df["rmse"], higher_is_better=False)

    # Generalisation gap — lower is better (0 = no overfitting)
    # Clip to [0, ∞) first — negative gap (model is better on validation than
    # train) is unusual but not penalised below zero.
    df["gen_gap_clipped"] = df["generalization_gap"].clip(lower=0.0)
    df["norm_gen_gap"]    = normalise_col(df["gen_gap_clipped"], higher_is_better=False)

    # Prediction time — lower is better
    df["norm_pred_time"] = normalise_col(df["val_pred_time_s"], higher_is_better=False)

    # Training time — lower is better
    df["norm_train_time"] = normalise_col(df["train_time_s"], higher_is_better=False)

    # Interpretability — already in [0, 1], higher is better
    df["norm_interp"] = df["interpretability"]

    # ── Weighted sum ───────────────────────────────────────────────────────────
    df["engineering_score"] = (
        _SCORE_WEIGHTS["validation_r2"]      * df["norm_val_r2"]    +
        _SCORE_WEIGHTS["test_rmse"]           * df["norm_test_rmse"] +
        _SCORE_WEIGHTS["generalization_gap"]  * df["norm_gen_gap"]   +
        _SCORE_WEIGHTS["prediction_time"]     * df["norm_pred_time"] +
        _SCORE_WEIGHTS["training_time"]       * df["norm_train_time"]+
        _SCORE_WEIGHTS["interpretability"]    * df["norm_interp"]
    )

    df["engineering_score"] = df["engineering_score"].round(6)

    logger.info("Engineering scores computed for %d models.", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEPLOYMENT STATUS
# ─────────────────────────────────────────────────────────────────────────────

def assign_deployment_status(row: pd.Series) -> str:
    """
    Assign a deployment tier to a single model row.

    Rules (evaluated in order — first match wins)
    ─────────────────────────────────────────────
    Production Ready   : Val R² ≥ 0.85  AND  gap ≤ 0.10
    Needs Optimization : Val R² ≥ 0.60  (but not Production Ready)
    Experimental       : everything else

    Status is derived entirely from metrics — no model names are hardcoded.

    Parameters
    ----------
    row : pd.Series (one row from the evaluation DataFrame)

    Returns
    -------
    str
    """
    val_r2  = row.get("validation_r2", np.nan)
    gap     = row.get("generalization_gap", np.nan)

    if np.isnan(val_r2):
        return "Experimental"

    if val_r2 >= _PROD_READY_MIN_VAL_R2:
        # Also check overfitting guard
        if np.isnan(gap) or gap <= _PROD_READY_MAX_GEN_GAP:
            return "Production Ready"
        else:
            return "Needs Optimization"

    if val_r2 >= _EXPERIMENTAL_MIN_VAL_R2:
        return "Needs Optimization"

    return "Experimental"


# ─────────────────────────────────────────────────────────────────────────────
# 6. RISK FLAGS
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_flags(row: pd.Series) -> List[str]:
    """
    Automatically detect performance and operational risk signals.

    Detected risks
    ──────────────
    Potential Overfitting    : gen gap > _RISK_OVERFIT_GAP
    Slow Prediction          : pred time > _RISK_SLOW_PRED_S seconds
    Slow Training            : train time > _RISK_SLOW_TRAIN_S seconds
    Weak Validation          : Val R² < _RISK_WEAK_VAL_R2
    High Generalisation Gap  : same as overfitting (displayed separately)
    Large Error              : RMSE > _RISK_HIGH_RMSE AQI units

    Parameters
    ----------
    row : pd.Series

    Returns
    -------
    list[str] — empty list if no risks detected
    """
    flags: List[str] = []

    gap      = row.get("generalization_gap", np.nan)
    pred_t   = row.get("val_pred_time_s",    np.nan)
    train_t  = row.get("train_time_s",       np.nan)
    val_r2   = row.get("validation_r2",      np.nan)
    rmse     = row.get("rmse",               np.nan)

    if not np.isnan(gap)    and gap    > _RISK_OVERFIT_GAP:
        flags.append("Potential Overfitting")
    if not np.isnan(pred_t) and pred_t > _RISK_SLOW_PRED_S:
        flags.append("Slow Prediction")
    if not np.isnan(train_t)and train_t> _RISK_SLOW_TRAIN_S:
        flags.append("Slow Training")
    if not np.isnan(val_r2) and val_r2 < _RISK_WEAK_VAL_R2:
        flags.append("Weak Validation Performance")
    if not np.isnan(gap)    and gap    > _RISK_OVERFIT_GAP:
        flags.append("High Generalisation Gap")
    if not np.isnan(rmse)   and rmse   > _RISK_HIGH_RMSE:
        flags.append("Large Error")

    # Deduplicate while preserving order
    seen:  set  = set()
    dedup: List[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            dedup.append(f)
    return dedup


# ─────────────────────────────────────────────────────────────────────────────
# 7. AI JUDGE COMMENTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_judge_comment(row: pd.Series) -> str:
    """
    Generate automatic plain-English commentary for one model.

    Comments are derived entirely from the model's metrics and name —
    no model names are hardcoded in the decision logic.

    Parameters
    ----------
    row : pd.Series (one row from the ranked leaderboard)

    Returns
    -------
    str — multi-sentence comment
    """
    name      = row["model"]
    val_r2    = row.get("validation_r2",      np.nan)
    gap       = row.get("generalization_gap",  np.nan)
    pred_t    = row.get("val_pred_time_s",     np.nan)
    interp    = row.get("interpretability",    0.5)
    status    = row.get("deployment_status",   "Experimental")
    rmse      = row.get("rmse",                np.nan)
    score     = row.get("engineering_score",   np.nan)

    sentences: List[str] = []

    # Accuracy assessment
    if not np.isnan(val_r2):
        if val_r2 >= 0.92:
            sentences.append("Outstanding validation accuracy.")
        elif val_r2 >= 0.85:
            sentences.append("Strong validation accuracy.")
        elif val_r2 >= 0.70:
            sentences.append("Moderate validation accuracy.")
        else:
            sentences.append("Below-threshold validation accuracy.")

    # Overfitting assessment
    if not np.isnan(gap):
        if gap <= 0.02:
            sentences.append("Minimal generalisation gap — excellent stability.")
        elif gap <= 0.10:
            sentences.append("Acceptable generalisation gap.")
        else:
            sentences.append(
                f"Generalisation gap of {gap:.3f} suggests overfitting — "
                "consider regularisation or early stopping."
            )

    # Prediction speed
    if not np.isnan(pred_t):
        if pred_t < 0.01:
            sentences.append("Near-instantaneous inference — ideal for real-time use.")
        elif pred_t < 0.5:
            sentences.append("Fast inference — suitable for production APIs.")
        else:
            sentences.append("Slower inference — may require optimisation for real-time.")

    # Interpretability
    if interp >= 0.90:
        sentences.append("Highly interpretable — excellent for policy reporting.")
    elif interp >= 0.60:
        sentences.append("Moderate interpretability — supports feature importance.")
    else:
        sentences.append("Lower interpretability — consider explainability tools.")

    # Error magnitude
    if not np.isnan(rmse):
        if rmse < 15:
            sentences.append(f"Low RMSE ({rmse:.1f} AQI units) — precise forecasts.")
        elif rmse < 35:
            sentences.append(f"Moderate RMSE ({rmse:.1f} AQI units).")
        else:
            sentences.append(f"High RMSE ({rmse:.1f} AQI units) — review feature set.")

    # Deployment conclusion
    if status == "Production Ready":
        sentences.append("Recommended for production deployment.")
    elif status == "Needs Optimization":
        sentences.append("Promising — further tuning recommended before deployment.")
    else:
        sentences.append("Experimental only — not suitable for production yet.")

    return "  ".join(sentences)


# ─────────────────────────────────────────────────────────────────────────────
# 8. LEADERBOARD ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_leaderboard(metrics_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Assemble, score, and rank the full leaderboard.

    Steps
    -----
    1. Flatten each metrics record into an evaluation row
    2. Compute generalisation gap
    3. Compute engineering scores (with normalisation)
    4. Assign deployment status
    5. Compute risk flags
    6. Generate AI judge comments
    7. Sort by engineering_score descending
    8. Assign rank

    Parameters
    ----------
    metrics_records : list[dict] from load_all_metrics()

    Returns
    -------
    pd.DataFrame (ranked leaderboard)
    """
    rows = [extract_evaluation_row(r) for r in metrics_records]
    df   = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid model metrics found — cannot build leaderboard.")

    # Compute engineering scores (adds norm_* and engineering_score columns)
    df = compute_engineering_scores(df)

    # Deployment status
    df["deployment_status"] = df.apply(assign_deployment_status, axis=1)

    # Risk flags as a pipe-separated string (JSON-friendly)
    df["risk_flags"] = df.apply(
        lambda row: " | ".join(compute_risk_flags(row)) or "None", axis=1
    )

    # AI judge comment
    df["ai_comment"] = df.apply(generate_judge_comment, axis=1)

    # Sort by engineering score descending
    df = df.sort_values("engineering_score", ascending=False).reset_index(drop=True)

    # Rank (1-indexed)
    df.insert(0, "rank", df.index + 1)

    logger.info("Leaderboard built — %d models ranked.", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. AI RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Automatically derive a deployment recommendation from the leaderboard.

    Logic
    ─────
    • Recommended model = rank 1 (highest engineering score)
    • Runner-up         = rank 2
    • Fastest prediction  = lowest val_pred_time_s
    • Most accurate       = highest validation_r2
    • Most interpretable  = highest interpretability score

    Recommendation reasons are constructed from the winning model's metrics —
    no model names are hardcoded in the logic.

    Parameters
    ----------
    df : ranked leaderboard DataFrame

    Returns
    -------
    dict with keys: recommended, runner_up, fastest_prediction,
                    most_accurate, most_interpretable, reasons
    """
    winner      = df.iloc[0]
    runner_up   = df.iloc[1] if len(df) > 1 else None
    fastest_pred = df.loc[df["val_pred_time_s"].idxmin()]
    most_accurate = df.loc[df["validation_r2"].idxmax()]
    most_interp   = df.loc[df["interpretability"].idxmax()]

    reasons: List[str] = []

    reasons.append(
        f"Highest Overall Engineering Score "
        f"({winner['engineering_score']:.4f} / 1.00)."
    )

    val_r2 = winner.get("validation_r2", np.nan)
    if not np.isnan(val_r2):
        reasons.append(f"Validation R² = {val_r2:.4f}.")

    gap = winner.get("generalization_gap", np.nan)
    if not np.isnan(gap):
        if gap <= 0.05:
            reasons.append("Negligible generalisation gap — robust to unseen data.")
        elif gap <= 0.10:
            reasons.append(f"Acceptable generalisation gap ({gap:.3f}).")
        else:
            reasons.append(f"Note: generalisation gap is {gap:.3f} — monitor in production.")

    pred_t = winner.get("val_pred_time_s", np.nan)
    if not np.isnan(pred_t):
        reasons.append(f"Prediction latency: {pred_t*1000:.1f} ms — suitable for real-time.")

    interp = winner.get("interpretability", 0.0)
    reasons.append(
        f"Interpretability score: {interp:.2f} "
        f"— {'supports' if interp >= 0.60 else 'limited'} feature importance."
    )

    status = winner.get("deployment_status", "Experimental")
    reasons.append(f"Deployment status: {status}.")

    return {
        "recommended_model":   winner["model"],
        "runner_up":           runner_up["model"] if runner_up is not None else "N/A",
        "fastest_prediction":  fastest_pred["model"],
        "most_accurate":       most_accurate["model"],
        "most_interpretable":  most_interp["model"],
        "reasons":             reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. LEADERBOARD SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_leaderboard(df: pd.DataFrame) -> None:
    """
    Save leaderboard as both CSV and JSON.

    Columns saved
    ─────────────
    rank, model, validation_r2, test_r2, mae, rmse, mape,
    train_time_s, val_pred_time_s, generalization_gap,
    engineering_score, deployment_status, risk_flags, ai_comment

    Parameters
    ----------
    df : ranked leaderboard DataFrame
    """
    os.makedirs(DIR_LEADERBOARD, exist_ok=True)

    export_cols = [
        "rank", "model", "validation_r2", "test_r2", "mae", "rmse", "mape",
        "train_time_s", "val_pred_time_s", "generalization_gap",
        "engineering_score", "deployment_status", "risk_flags", "ai_comment",
    ]
    # Only include columns that exist in df
    export_cols = [c for c in export_cols if c in df.columns]
    export_df   = df[export_cols].copy()

    # CSV
    export_df.to_csv(PATH_LEADERBOARD_CSV, index=False)
    logger.info("Leaderboard CSV saved → %s", PATH_LEADERBOARD_CSV)

    # JSON (list of records)
    export_df.to_json(
        PATH_LEADERBOARD_JSON, orient="records", indent=2, force_ascii=False
    )
    logger.info("Leaderboard JSON saved → %s", PATH_LEADERBOARD_JSON)


# ─────────────────────────────────────────────────────────────────────────────
# 11. CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def _bar_chart(
    names:      List[str],
    values:     List[float],
    title:      str,
    xlabel:     str,
    ylabel:     str,
    filename:   str,
    higher_is_better: bool = True,
) -> None:
    """
    Render and save a horizontal bar chart with one bar per model.

    The best-performing bar is highlighted in a distinct colour.
    All other rendering details (size, font, colours) use the module-level
    _FIG_* and _BAR_COLOR / _BEST_COLOR constants.

    Parameters
    ----------
    names            : model names (y-axis labels)
    values           : metric values (x-axis)
    title            : chart title
    xlabel           : x-axis label
    ylabel           : y-axis label
    filename         : output filename (placed in DIR_CHARTS)
    higher_is_better : controls which bar receives the highlight colour
    """
    os.makedirs(DIR_CHARTS, exist_ok=True)
    path = os.path.join(DIR_CHARTS, filename)

    # Determine best bar index
    valid = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
    if not valid:
        logger.warning("No valid values for chart '%s' — skipping.", title)
        return
    best_idx = max(valid, key=lambda x: x[1])[0] if higher_is_better \
               else min(valid, key=lambda x: x[1])[0]

    colours = [_BEST_COLOR if i == best_idx else _BAR_COLOR
               for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, max(_FIG_HEIGHT, len(names) * 0.55)))
    y_pos   = np.arange(len(names))

    ax.barh(y_pos, values, color=colours, edgecolor="white", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=_FONT_AXIS)
    ax.invert_yaxis()           # best model at the top
    ax.set_xlabel(xlabel, fontsize=_FONT_AXIS)
    ax.set_ylabel(ylabel, fontsize=_FONT_AXIS)
    ax.set_title(title, fontsize=_FONT_TITLE, fontweight="bold", pad=14)

    # Annotate each bar with its value
    for i, v in enumerate(values):
        if not np.isnan(v):
            ax.text(
                v * 1.005, i, f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}",
                va="center", fontsize=9, color="#333333",
            )

    # Legend patch for best bar
    from matplotlib.patches import Patch
    legend_label = "Best" if higher_is_better else "Best (lowest)"
    ax.legend(
        handles=[Patch(color=_BEST_COLOR, label=legend_label)],
        loc="lower right", fontsize=9,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", path)


def generate_all_charts(df: pd.DataFrame) -> None:
    """
    Generate all comparison charts from the ranked leaderboard.

    Charts produced
    ───────────────
    validation_r2.png      — Val R² per model (higher = better)
    test_rmse.png          — Test RMSE per model (lower = better)
    mae.png                — Validation MAE (lower = better)
    mape.png               — Validation MAPE % (lower = better)
    training_time.png      — Training wall-clock time (lower = better)
    prediction_time.png    — Validation prediction time (lower = better)
    generalization_gap.png — Train R² − Val R² (lower = better)
    engineering_score.png  — Overall Engineering Score (higher = better)

    Parameters
    ----------
    df : ranked leaderboard DataFrame
    """
    logger.info("Generating comparison charts …")

    names = df["model"].tolist()

    def _vals(col: str) -> List[float]:
        return df[col].fillna(np.nan).tolist()

    _bar_chart(names, _vals("validation_r2"), "Validation R² by Model",
               "R²", "Model", "validation_r2.png", higher_is_better=True)

    _bar_chart(names, _vals("rmse"), "Validation RMSE by Model",
               "RMSE (AQI units)", "Model", "test_rmse.png", higher_is_better=False)

    _bar_chart(names, _vals("mae"), "Validation MAE by Model",
               "MAE (AQI units)", "Model", "mae.png", higher_is_better=False)

    _bar_chart(names, _vals("mape"), "Validation MAPE by Model",
               "MAPE (%)", "Model", "mape.png", higher_is_better=False)

    _bar_chart(names, _vals("train_time_s"), "Training Time by Model",
               "Seconds", "Model", "training_time.png", higher_is_better=False)

    _bar_chart(names, _vals("val_pred_time_s"), "Prediction Time by Model",
               "Seconds", "Model", "prediction_time.png", higher_is_better=False)

    _bar_chart(names, _vals("generalization_gap"),
               "Generalisation Gap (Train R² − Val R²) by Model",
               "Gap", "Model", "generalization_gap.png", higher_is_better=False)

    _bar_chart(names, _vals("engineering_score"), "Overall Engineering Score by Model",
               "Score (0 – 1)", "Model", "engineering_score.png", higher_is_better=True)

    logger.info("All charts generated.")


# ─────────────────────────────────────────────────────────────────────────────
# 12. EXECUTIVE SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_executive_summary(
    df:             pd.DataFrame,
    recommendation: Dict[str, Any],
) -> str:
    """
    Compose the one-page executive summary.

    Parameters
    ----------
    df             : ranked leaderboard DataFrame
    recommendation : output of generate_recommendation()

    Returns
    -------
    str
    """
    sep  = "=" * 70
    sep2 = "-" * 70
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prod_ready = df[df["deployment_status"] == "Production Ready"]["model"].tolist()
    n_models   = len(df)

    winner     = df.iloc[0]
    runner_up  = df.iloc[1] if len(df) > 1 else None

    fastest_pred_row  = df.loc[df["val_pred_time_s"].idxmin()]
    most_accurate_row = df.loc[df["validation_r2"].idxmax()]
    most_interp_row   = df.loc[df["interpretability"].idxmax()]

    lines = [
        sep,
        "  AIRSENSE AI – MODEL EVALUATION EXECUTIVE SUMMARY",
        f"  Generated : {ts}",
        "  Script    : utils/05_model_comparison.py",
        sep,
        "",
        "[1] EVALUATION OVERVIEW",
        sep2,
        f"  Models evaluated          : {n_models}",
        f"  Production Ready models   : {len(prod_ready)}  "
        f"({', '.join(prod_ready) or 'None'})",
        f"  Score weights             : "
        f"ValR²={_SCORE_WEIGHTS['validation_r2']*100:.0f}%  "
        f"RMSE={_SCORE_WEIGHTS['test_rmse']*100:.0f}%  "
        f"Gap={_SCORE_WEIGHTS['generalization_gap']*100:.0f}%  "
        f"PredT={_SCORE_WEIGHTS['prediction_time']*100:.0f}%  "
        f"TrainT={_SCORE_WEIGHTS['training_time']*100:.0f}%  "
        f"Interp={_SCORE_WEIGHTS['interpretability']*100:.0f}%",
        "",
        "[2] TOP PERFORMERS",
        sep2,
        f"  🥇 Winner              : {winner['model']}",
        f"     Engineering Score   : {winner['engineering_score']:.4f}",
        f"     Validation R²       : {winner['validation_r2']:.4f}",
        f"     Validation RMSE     : {winner['rmse']:.2f} AQI units",
        f"     Deployment Status   : {winner['deployment_status']}",
    ]

    if runner_up is not None:
        lines += [
            "",
            f"  🥈 Runner-Up           : {runner_up['model']}",
            f"     Engineering Score   : {runner_up['engineering_score']:.4f}",
            f"     Validation R²       : {runner_up['validation_r2']:.4f}",
        ]

    lines += [
        "",
        f"  ⚡ Fastest Prediction   : {fastest_pred_row['model']}  "
        f"({fastest_pred_row['val_pred_time_s']*1000:.2f} ms)",
        f"  🎯 Most Accurate        : {most_accurate_row['model']}  "
        f"(Val R² = {most_accurate_row['validation_r2']:.4f})",
        f"  📖 Most Interpretable   : {most_interp_row['model']}  "
        f"(score = {most_interp_row['interpretability']:.2f})",
        "",
        "[3] DEPLOYMENT RECOMMENDATION",
        sep2,
        f"  Recommended Model : {recommendation['recommended_model']}",
        "",
        "  Reasoning:",
    ]

    for reason in recommendation["reasons"]:
        lines.append(f"    •  {reason}")

    lines += [
        "",
        "[4] OUTPUT FILES",
        sep2,
        f"  Leaderboard CSV   : {PATH_LEADERBOARD_CSV}",
        f"  Leaderboard JSON  : {PATH_LEADERBOARD_JSON}",
        f"  Full report       : {PATH_EVAL_REPORT}",
        f"  Charts directory  : {DIR_CHARTS}/",
        "",
        sep,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 13. FULL EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_full_report(
    df:             pd.DataFrame,
    recommendation: Dict[str, Any],
    metadata:       Dict[str, Dict[str, Any]],
) -> str:
    """
    Compose the detailed per-model evaluation report.

    Sections
    ────────
    1. Leaderboard table
    2. Score weight methodology
    3. Per-model detailed analysis (metrics + strengths + weaknesses + risks)
    4. AI judge comments
    5. Deployment summary table
    6. Recommendation

    Parameters
    ----------
    df             : ranked leaderboard DataFrame
    recommendation : output of generate_recommendation()
    metadata       : model metadata dict keyed by model name

    Returns
    -------
    str
    """
    sep  = "=" * 76
    sep2 = "-" * 76
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        sep,
        "  AIRSENSE AI – FULL MODEL EVALUATION REPORT",
        f"  Generated : {ts}",
        "  Script    : utils/05_model_comparison.py",
        sep,
        "",
        "[1] LEADERBOARD",
        sep2,
        f"  {'Rank':<5} {'Model':<22} {'ValR²':>7} {'TestR²':>7} "
        f"{'MAE':>7} {'RMSE':>8} {'MAPE%':>7} "
        f"{'Gap':>7} {'Score':>8} {'Status'}",
        sep2,
    ]

    for _, row in df.iterrows():
        lines.append(
            f"  {int(row['rank']):<5} "
            f"{str(row['model']):<22} "
            f"{row['validation_r2']:>7.4f} "
            f"{row['test_r2']:>7.4f} "
            f"{row['mae']:>7.2f} "
            f"{row['rmse']:>8.2f} "
            f"{row['mape']:>7.2f} "
            f"{row['generalization_gap']:>7.4f} "
            f"{row['engineering_score']:>8.4f} "
            f"{row['deployment_status']}"
        )

    lines += [
        "",
        "[2] ENGINEERING SCORE METHODOLOGY",
        sep2,
        "  The Engineering Score is a weighted composite of six normalised",
        "  sub-scores, each in [0, 1].  Normalisation uses min-max scaling",
        "  across all models so the score is relative to the current cohort.",
        "",
        f"  {'Component':<26} {'Weight':>8}   {'Direction'}",
        sep2,
    ]

    component_info = [
        ("Validation R²",       _SCORE_WEIGHTS["validation_r2"],      "Higher is better"),
        ("Test RMSE",           _SCORE_WEIGHTS["test_rmse"],           "Lower is better"),
        ("Generalisation Gap",  _SCORE_WEIGHTS["generalization_gap"],  "Lower is better"),
        ("Prediction Time",     _SCORE_WEIGHTS["prediction_time"],     "Lower is better"),
        ("Training Time",       _SCORE_WEIGHTS["training_time"],       "Lower is better"),
        ("Interpretability",    _SCORE_WEIGHTS["interpretability"],    "Higher is better"),
    ]
    for comp, w, direction in component_info:
        lines.append(f"  {comp:<26} {w*100:>7.0f}%   {direction}")

    lines += ["", "[3] PER-MODEL DETAILED ANALYSIS", sep2]

    for _, row in df.iterrows():
        model  = row["model"]
        rank   = int(row["rank"])
        status = row["deployment_status"]
        risks  = row["risk_flags"]
        comment = row["ai_comment"]

        # Pull hyperparameters from metadata if available
        meta   = metadata.get(model, {})
        hparams = meta.get("hyperparameters", {})
        n_feat  = meta.get("n_features", "N/A")
        t_rows  = meta.get("training_rows", "N/A")

        lines += [
            "",
            f"  ┌─ Rank {rank}: {model}  [{status}]  {'─' * max(1, 44 - len(model))}",
            f"  │  Validation R²        : {row['validation_r2']:.4f}",
            f"  │  Test R²              : {row['test_r2']:.4f}",
            f"  │  Validation MAE       : {row['mae']:.2f} AQI units",
            f"  │  Validation RMSE      : {row['rmse']:.2f} AQI units",
            f"  │  Validation MAPE      : {row['mape']:.2f}%",
            f"  │  Generalisation Gap   : {row['generalization_gap']:.4f}",
            f"  │  Training Time        : {row['train_time_s']:.2f} s",
            f"  │  Prediction Time      : {row['val_pred_time_s']*1000:.2f} ms",
            f"  │  Interpretability     : {row['interpretability']:.2f}",
            f"  │  Engineering Score    : {row['engineering_score']:.4f}",
            f"  │  Data used (scaled?)  : {row.get('uses_scaled', 'N/A')}",
            f"  │  Features             : {n_feat}",
            f"  │  Training rows        : {t_rows}",
            f"  │",
            f"  │  Risk Flags    : {risks}",
            f"  │",
            f"  │  AI Comment    : {comment}",
        ]

        if hparams:
            lines.append(f"  │  Key Hyperparameters:")
            for k, v in list(hparams.items())[:6]:   # show first 6 only
                lines.append(f"  │    {k:<30} : {v}")

        lines.append(f"  └{'─' * 60}")

    lines += [
        "",
        "[4] DEPLOYMENT SUMMARY",
        sep2,
        f"  {'Model':<22} {'Status':<22} {'Risk Flags'}",
        sep2,
    ]

    for _, row in df.iterrows():
        lines.append(
            f"  {str(row['model']):<22} "
            f"{str(row['deployment_status']):<22} "
            f"{row['risk_flags']}"
        )

    lines += [
        "",
        "[5] RECOMMENDATION",
        sep2,
        f"  Recommended Model : {recommendation['recommended_model']}",
        "",
        "  Reasoning:",
    ]
    for reason in recommendation["reasons"]:
        lines.append(f"    •  {reason}")

    lines += [
        "",
        sep,
        "  Full evaluation complete.  Next stage: dashboard / deployment.",
        sep,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 14. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Execute the full model comparison and evaluation pipeline.

    Steps
    -----
    1.  Load all model metrics (metrics/*.json)
    2.  Load all model metadata (models_metadata/*.json)
    3.  Build and score the leaderboard
    4.  Save leaderboard CSV + JSON
    5.  Generate AI recommendation
    6.  Generate all comparison charts
    7.  Write executive summary report
    8.  Write full evaluation report
    """
    import time
    t0 = time.perf_counter()

    logger.info("=" * 76)
    logger.info("  AirSense AI — Model Comparison & Evaluation Pipeline START")
    logger.info("=" * 76)

    # ── Step 1: Load metrics ──────────────────────────────────────────────────
    logger.info("Loading model metrics from: %s/", DIR_METRICS)
    metrics_records = load_all_metrics(DIR_METRICS)

    if not metrics_records:
        raise ValueError(
            f"No valid metric files found in '{DIR_METRICS}/'. "
            "Run utils/04_train_models.py first."
        )

    # ── Step 2: Load metadata ─────────────────────────────────────────────────
    logger.info("Loading model metadata from: %s/", DIR_METADATA)
    metadata = load_all_metadata(DIR_METADATA)

    # ── Step 3: Build leaderboard ─────────────────────────────────────────────
    logger.info("Building leaderboard …")
    leaderboard = build_leaderboard(metrics_records)

    # ── Step 4: Save leaderboard ──────────────────────────────────────────────
    logger.info("Saving leaderboard …")
    save_leaderboard(leaderboard)

    # ── Step 5: AI recommendation ─────────────────────────────────────────────
    logger.info("Generating AI recommendation …")
    recommendation = generate_recommendation(leaderboard)
    logger.info(
        "Recommended model: %s  (score=%.4f)",
        recommendation["recommended_model"],
        leaderboard.iloc[0]["engineering_score"],
    )

    # ── Step 6: Charts ────────────────────────────────────────────────────────
    logger.info("Generating comparison charts …")
    try:
        generate_all_charts(leaderboard)
    except Exception as exc:
        logger.warning("Chart generation encountered an error: %s", exc)

    # ── Step 7: Executive summary ─────────────────────────────────────────────
    logger.info("Writing executive summary …")
    os.makedirs(DIR_REPORTS, exist_ok=True)
    summary_text = build_executive_summary(leaderboard, recommendation)
    with open(PATH_EVAL_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(summary_text)
    logger.info("Executive summary saved → %s", PATH_EVAL_SUMMARY)

    # ── Step 8: Full evaluation report ───────────────────────────────────────
    logger.info("Writing full evaluation report …")
    full_report_text = build_full_report(leaderboard, recommendation, metadata)
    with open(PATH_EVAL_REPORT, "w", encoding="utf-8") as fh:
        fh.write(full_report_text)
    logger.info("Full evaluation report saved → %s", PATH_EVAL_REPORT)

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print(summary_text)
    print()

    elapsed = time.perf_counter() - t0
    logger.info("=" * 76)
    logger.info(
        "  AirSense AI — Model Comparison Pipeline COMPLETE (%.2f s)", elapsed
    )
    logger.info("=" * 76)


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
    except ValueError as exc:
        logger.error("Data error: %s", exc)
        raise SystemExit(1) from exc
    except PermissionError as exc:
        logger.error("Permission denied: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error during model comparison: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()