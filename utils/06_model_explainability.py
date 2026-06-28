"""
AirSense AI – Urban Air Quality Intelligence Platform
=====================================================
Script  : utils/06_model_explainability.py
Purpose : Explain the single best model selected by 05_model_comparison.py.

          Reads leaderboard/leaderboard.json, identifies the rank-1 model,
          loads only that model artefact, extracts feature importance using
          the method appropriate for the model family, and writes a focused
          set of outputs for reporting and dashboard consumption.

          Design principle
          ─────────────────
          One model. One explanation. No iteration over all saved models.
          The leaderboard already ranks every candidate; this script drills
          into the winner exclusively.

          Model family support
          ────────────────────
          Tree / Boosting   │ .feature_importances_
          ──────────────────┼──────────────────────────────────────────────
          Linear            │ |.coef_|  (absolute value used for ranking;
                            │  signed value preserved for interpretation)
          ──────────────────┼──────────────────────────────────────────────
          MLP / Unknown     │ Warning logged; placeholder outputs written;
                            │ pipeline continues without crashing.
                            │ SHAP support planned for Version 2.

          Outputs
          ───────
          reports/model_explainability_report.txt
          charts/feature_importance.png
          charts/top10_features.png
          charts/pollutant_importance.png
          explainability/feature_importance.csv
          explainability/feature_importance.json
          explainability/dashboard_explainability.json

Author  : AirSense AI Engineering Team
Python  : 3.11+
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on servers / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Path constants ─────────────────────────────────────────────────────────────
PATH_LEADERBOARD   = os.path.join("leaderboard", "leaderboard.json")
DIR_MODELS         = "models"
PATH_TRAIN_CSV     = os.path.join("datasets", "prepared", "train.csv")

DIR_REPORTS        = "reports"
DIR_CHARTS         = "charts"
DIR_EXPLAINABILITY = "explainability"

PATH_REPORT      = os.path.join(DIR_REPORTS,        "model_explainability_report.txt")
PATH_CHART_ALL   = os.path.join(DIR_CHARTS,         "feature_importance.png")
PATH_CHART_TOP10 = os.path.join(DIR_CHARTS,         "top10_features.png")
PATH_CHART_POLL  = os.path.join(DIR_CHARTS,         "pollutant_importance.png")
PATH_CSV         = os.path.join(DIR_EXPLAINABILITY, "feature_importance.csv")
PATH_JSON_IMP    = os.path.join(DIR_EXPLAINABILITY, "feature_importance.json")
PATH_JSON_DASH   = os.path.join(DIR_EXPLAINABILITY, "dashboard_explainability.json")

# ── Domain constants ───────────────────────────────────────────────────────────
TARGET_COL = "AQI"

# Columns excluded from the feature matrix (mirrors 04_train_models.py)
DROP_COLS: List[str] = [
    "City", "Date", "AQI_Bucket", "Season",
    "temperature", "humidity", "wind_speed", "rainfall",
]

# Keywords used to identify pollutant-related features (partial, case-insensitive)
POLLUTANT_KEYWORDS: List[str] = [
    "PM2.5", "PM10", "NO2", "NO", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene",
]

# ── Chart style tokens ─────────────────────────────────────────────────────────
_BG     = "#0b1630"   # dark navy background
_FG     = "#f0f6ff"   # near-white text
_GREEN  = "#39d353"   # standard bar colour
_BRIGHT = "#6ee87a"   # highlight colour for the top-ranked bar
_GRID   = "#1e2f54"   # subtle gridline colour
_WARN   = "#f5a623"   # amber used in placeholder / warning charts


# ─────────────────────────────────────────────────────────────────────────────
# 1. DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────

def ensure_output_dirs() -> None:
    """Create every required output directory (idempotent)."""
    for directory in [DIR_REPORTS, DIR_CHARTS, DIR_EXPLAINABILITY]:
        os.makedirs(directory, exist_ok=True)
    logger.info("Output directories verified.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LEADERBOARD — READ & IDENTIFY BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

def read_leaderboard(path: str) -> List[Dict[str, Any]]:
    """
    Load and perform basic validation on the leaderboard JSON.

    Parameters
    ----------
    path : str
        Path to leaderboard/leaderboard.json.

    Returns
    -------
    list[dict]
        Ordered list of model records as written by 05_model_comparison.py.

    Raises
    ------
    FileNotFoundError
        When the file is absent — caller should run 05_model_comparison.py.
    ValueError
        When the JSON root is not a non-empty list.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Leaderboard not found at '{path}'. "
            "Run utils/05_model_comparison.py first."
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"Leaderboard at '{path}' is empty or not a JSON array. "
            "Re-run utils/05_model_comparison.py."
        )

    logger.info("Leaderboard loaded — %d model records found.", len(data))
    return data


def get_best_model_record(leaderboard: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return the rank-1 record from the leaderboard.

    When a 'rank' field exists the list is sorted ascending by rank before
    selecting; otherwise the first element is assumed to be rank 1.

    Parameters
    ----------
    leaderboard : list[dict]
        Raw leaderboard data from read_leaderboard().

    Returns
    -------
    dict
        The single best-model record.

    Raises
    ------
    ValueError
        When no valid model name can be found in the top record.
    """
    if "rank" in leaderboard[0]:
        leaderboard = sorted(leaderboard, key=lambda r: int(r.get("rank", 9999)))

    best = leaderboard[0]

    if not best.get("model"):
        raise ValueError(
            "The rank-1 leaderboard entry has no 'model' field. "
            "Re-run utils/05_model_comparison.py."
        )

    logger.info(
        "Best model → '%s'  "
        "(Engineering Score=%.4f | Validation R²=%.4f | Status=%s)",
        best["model"],
        best.get("engineering_score", float("nan")),
        best.get("validation_r2",     float("nan")),
        best.get("deployment_status", "N/A"),
    )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, models_dir: str) -> Any:
    """
    Deserialise a trained model from its .joblib artefact.

    Parameters
    ----------
    model_name  : str
        Stem of the .joblib file (no extension).
    models_dir  : str
        Directory that contains the .joblib files.

    Returns
    -------
    Any
        Fitted sklearn-compatible estimator.

    Raises
    ------
    FileNotFoundError
        When the .joblib file is missing — caller should run 04_train_models.py.
    RuntimeError
        When joblib cannot deserialise the file.
    """
    path = os.path.join(models_dir, f"{model_name}.joblib")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model artefact not found: '{path}'. "
            "Run utils/04_train_models.py first."
        )

    try:
        model = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load '{model_name}' from '{path}': {exc}"
        ) from exc

    logger.info("Model '%s' loaded from: %s", model_name, path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE LIST FROM TRAINING CSV
# ─────────────────────────────────────────────────────────────────────────────

def read_feature_columns(train_csv: str) -> List[str]:
    """
    Derive the ordered feature column list by reading only the CSV header.

    Applies the same drop-logic used in 04_train_models.py so the feature
    list is always consistent with what the model was trained on.

    Parameters
    ----------
    train_csv : str
        Path to datasets/prepared/train.csv.

    Returns
    -------
    list[str]
        Ordered list of numeric feature column names.

    Raises
    ------
    FileNotFoundError
        When the CSV is absent.
    """
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(
            f"Training CSV not found: '{train_csv}'. "
            "Run utils/03_data_preparation.py first."
        )

    # nrows=0 reads only the header — zero memory overhead
    header = pd.read_csv(train_csv, nrows=0)
    excluded = set(DROP_COLS) | {TARGET_COL}

    # Identify numeric columns from a tiny sample because dtype is
    # unknown from the header alone
    sample = pd.read_csv(train_csv, nrows=5)
    numeric_cols = set(sample.select_dtypes(include=[np.number]).columns)

    feature_cols = [
        c for c in header.columns
        if c not in excluded and c in numeric_cols
    ]

    logger.info(
        "Feature list built — %d features from: %s", len(feature_cols), train_csv
    )
    return feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL FAMILY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_family(model: Any) -> str:
    """
    Determine the model family by inspecting available attributes.

    Priority order
    ──────────────
    1. ``feature_importances_`` → ``"tree"``
    2. ``coef_``                → ``"linear"``
    3. (fallback)               → ``"unsupported"``

    Parameters
    ----------
    model : Any
        Fitted estimator.

    Returns
    -------
    str
        One of ``"tree"``, ``"linear"``, or ``"unsupported"``.
    """
    if hasattr(model, "feature_importances_"):
        return "tree"
    if hasattr(model, "coef_"):
        return "linear"
    return "unsupported"


# ─────────────────────────────────────────────────────────────────────────────
# 6. IMPORTANCE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _check_length(
    values:       np.ndarray,
    feature_cols: List[str],
    attr:         str,
    model_name:   str,
) -> bool:
    """
    Verify that *values* and *feature_cols* have the same length.

    Logs an actionable error message if they differ and returns False so the
    caller can return None instead of propagating a hard crash.

    Parameters
    ----------
    values       : importance / coefficient array
    feature_cols : expected feature list
    attr         : attribute name being checked (for the log message)
    model_name   : for the log message

    Returns
    -------
    bool
    """
    if len(values) != len(feature_cols):
        logger.error(
            "[%s] Length of %s (%d) does not match feature_cols (%d). "
            "Verify that train.csv matches the dataset used during training.",
            model_name, attr, len(values), len(feature_cols),
        )
        return False
    return True


def _attach_pct_columns(df: pd.DataFrame, raw_col: str) -> pd.DataFrame:
    """
    Append ``importance_pct`` and ``cumulative_pct`` to *df*.

    *df* must already be sorted descending by *raw_col* so that the
    cumulative sum is monotonically increasing from rank 1 downward.

    Parameters
    ----------
    df      : DataFrame to modify in-place (a copy is returned).
    raw_col : name of the column whose values sum to the total.

    Returns
    -------
    pd.DataFrame
    """
    total = df[raw_col].sum()
    df["importance_pct"] = (
        (df[raw_col] / total * 100).round(4) if total > 0
        else 0.0
    )
    df["cumulative_pct"] = df["importance_pct"].cumsum().round(4)
    return df


def extract_tree_importance(
    model:        Any,
    feature_cols: List[str],
    model_name:   str,
) -> Optional[pd.DataFrame]:
    """
    Build a ranked importance DataFrame from ``model.feature_importances_``.

    Output columns
    ──────────────
    rank | feature | importance | importance_pct | cumulative_pct

    Parameters
    ----------
    model        : tree or boosting estimator
    feature_cols : ordered feature names
    model_name   : for log messages

    Returns
    -------
    pd.DataFrame or None when extraction fails
    """
    raw = np.asarray(model.feature_importances_, dtype=float)

    if not _check_length(raw, feature_cols, "feature_importances_", model_name):
        return None

    df = (
        pd.DataFrame({"feature": feature_cols, "importance": raw})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", df.index + 1)
    df = _attach_pct_columns(df, "importance")

    logger.info(
        "[%s] Tree importance extracted — "
        "top feature: '%s' (%.2f%% of total).",
        model_name, df.iloc[0]["feature"], df.iloc[0]["importance_pct"],
    )
    return df


def extract_linear_importance(
    model:        Any,
    feature_cols: List[str],
    model_name:   str,
) -> Optional[pd.DataFrame]:
    """
    Build a ranked importance DataFrame from ``|model.coef_|``.

    The signed coefficient is preserved in the ``coefficient`` column for
    directional interpretation (positive → raises AQI; negative → lowers it).
    The absolute value drives ranking and percentage computation, and is also
    aliased to ``importance`` so chart helpers work without branching.

    Output columns
    ──────────────
    rank | feature | coefficient | absolute_coefficient
         | importance | importance_pct | cumulative_pct

    Parameters
    ----------
    model        : linear estimator
    feature_cols : ordered feature names
    model_name   : for log messages

    Returns
    -------
    pd.DataFrame or None when extraction fails
    """
    coef     = np.asarray(model.coef_, dtype=float)
    abs_coef = np.abs(coef)

    if not _check_length(coef, feature_cols, "coef_", model_name):
        return None

    df = (
        pd.DataFrame({
            "feature":              feature_cols,
            "coefficient":          coef,
            "absolute_coefficient": abs_coef,
        })
        .sort_values("absolute_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", df.index + 1)
    df["importance"] = df["absolute_coefficient"]   # unified alias for charts
    df = _attach_pct_columns(df, "absolute_coefficient")

    logger.info(
        "[%s] Linear coefficient importance extracted — "
        "top feature: '%s' (|coef|=%.4f, %.2f%% of total).",
        model_name,
        df.iloc[0]["feature"],
        df.iloc[0]["absolute_coefficient"],
        df.iloc[0]["importance_pct"],
    )
    return df


def extract_importance(
    model:        Any,
    family:       str,
    feature_cols: List[str],
    model_name:   str,
) -> Optional[pd.DataFrame]:
    """
    Dispatch feature importance extraction to the correct handler.

    Parameters
    ----------
    model        : fitted estimator
    family       : ``"tree"`` | ``"linear"`` | ``"unsupported"``
    feature_cols : ordered feature names
    model_name   : for log messages

    Returns
    -------
    pd.DataFrame or None
        None is returned (with an appropriate warning) for unsupported families.
    """
    if family == "tree":
        return extract_tree_importance(model, feature_cols, model_name)

    if family == "linear":
        return extract_linear_importance(model, feature_cols, model_name)

    # MLP or any other unsupported family
    logger.warning(
        "[%s] Intrinsic feature importance unavailable. "
        "SHAP support planned for Version 2.",
        model_name,
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. POLLUTANT FEATURE FILTER
# ─────────────────────────────────────────────────────────────────────────────

def filter_pollutant_rows(
    imp_df:   pd.DataFrame,
    keywords: List[str],
) -> pd.DataFrame:
    """
    Return rows whose 'feature' name contains any pollutant keyword.

    Matching is case-insensitive and partial so lag variants such as
    ``PM2.5_lag_1`` or ``AQI_roll_mean_7`` are captured automatically.
    Returned rows are re-ranked from 1.

    Parameters
    ----------
    imp_df   : full importance DataFrame
    keywords : pollutant keyword strings

    Returns
    -------
    pd.DataFrame — subset of *imp_df*, re-ranked
    """
    mask = imp_df["feature"].apply(
        lambda name: any(kw.lower() in name.lower() for kw in keywords)
    )
    subset = imp_df[mask].copy().reset_index(drop=True)
    subset["rank"] = subset.index + 1
    return subset


# ─────────────────────────────────────────────────────────────────────────────
# 8. CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_style(ax: plt.Axes, fig: plt.Figure) -> None:
    """
    Apply the AirSense AI dark-navy / green visual style to one axes object.

    Parameters
    ----------
    ax  : target Axes
    fig : parent Figure
    """
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, labelsize=9)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(axis="x", color=_GRID, linestyle="--", alpha=0.45)


def _save_bar_chart(
    features:   List[str],
    values:     List[float],
    pcts:       List[float],
    title:      str,
    xlabel:     str,
    out_path:   str,
) -> None:
    """
    Render and save a horizontal bar chart with percentage annotations.

    The bar for the feature with the highest importance (last element after
    ascending sort for barh) is rendered in a brighter green to draw
    attention to the top predictor.

    Parameters
    ----------
    features  : y-axis labels (feature names, in ascending importance order)
    values    : bar lengths (raw importance in the same order as *features*)
    pcts      : percentage labels annotated to the right of each bar
    title     : chart title string
    xlabel    : x-axis label
    out_path  : absolute or relative path for the saved PNG
    """
    n       = len(features)
    colours = [_BRIGHT if i == n - 1 else _GREEN for i in range(n)]

    fig, ax = plt.subplots(figsize=(13, max(5, n * 0.52)))
    _apply_style(ax, fig)

    y_pos = np.arange(n)
    ax.barh(y_pos, values, color=colours, edgecolor=_GRID, height=0.68)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9, color=_FG)
    ax.invert_yaxis()     # highest importance at the top

    for i, (v, p) in enumerate(zip(values, pcts)):
        ax.text(
            v * 1.005, i,
            f"{p:.2f}%",
            va="center", ha="left",
            color=_FG, fontsize=8,
        )

    ax.set_xlabel(xlabel, fontsize=11, color=_FG)
    ax.set_title(title, fontsize=13, color=_FG, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    logger.info("Chart saved → %s", out_path)


def _imp_series(
    df: pd.DataFrame,
) -> Tuple[List[str], List[float], List[float]]:
    """
    Extract (features, importance_values, importance_pcts) in ascending order
    so that ``ax.barh`` displays the highest-ranked feature at the top.

    Parameters
    ----------
    df : importance DataFrame (sorted descending by rank)

    Returns
    -------
    Tuple of three lists in ascending importance order.
    """
    features = df["feature"].tolist()[::-1]
    values   = df["importance"].tolist()[::-1]
    pcts     = df["importance_pct"].tolist()[::-1]
    return features, values, pcts


def _method_label(family: str) -> str:
    """Return a short human-readable importance method label."""
    return (
        "feature_importances_" if family == "tree"
        else "|coefficient|"   if family == "linear"
        else "N/A"
    )


def plot_all_features(
    imp_df:     pd.DataFrame,
    model_name: str,
    family:     str,
) -> None:
    """
    Chart 1 — All features ranked by importance.

    Parameters
    ----------
    imp_df     : full importance DataFrame
    model_name : displayed in the chart title
    family     : used to choose the method label
    """
    features, values, pcts = _imp_series(imp_df)
    _save_bar_chart(
        features = features,
        values   = values,
        pcts     = pcts,
        title    = (
            f"{model_name} — All Feature Importances\n"
            f"Method: {_method_label(family)}"
        ),
        xlabel   = "Importance Value",
        out_path = PATH_CHART_ALL,
    )


def plot_top10(
    imp_df:     pd.DataFrame,
    model_name: str,
    family:     str,
) -> None:
    """
    Chart 2 — Top-10 features ranked by importance.

    Parameters
    ----------
    imp_df     : full importance DataFrame (top 10 rows are used)
    model_name : displayed in the chart title
    family     : used to choose the method label
    """
    features, values, pcts = _imp_series(imp_df.head(10))
    _save_bar_chart(
        features = features,
        values   = values,
        pcts     = pcts,
        title    = (
            f"{model_name} — Top 10 Feature Importances\n"
            f"Method: {_method_label(family)}"
        ),
        xlabel   = "Importance Value",
        out_path = PATH_CHART_TOP10,
    )


def plot_pollutants(
    poll_df:    pd.DataFrame,
    model_name: str,
    family:     str,
) -> None:
    """
    Chart 3 — Pollutant-only features ranked by importance.

    Skipped (with a warning) when *poll_df* is empty so the pipeline never
    crashes on datasets that lack pollutant columns.

    Parameters
    ----------
    poll_df    : pollutant-filtered importance DataFrame
    model_name : displayed in the chart title
    family     : used to choose the method label
    """
    if poll_df.empty:
        logger.warning(
            "No pollutant features found in importance results — "
            "pollutant chart skipped."
        )
        return

    features, values, pcts = _imp_series(poll_df)
    _save_bar_chart(
        features = features,
        values   = values,
        pcts     = pcts,
        title    = (
            f"{model_name} — Pollutant Feature Importances\n"
            f"Method: {_method_label(family)}"
        ),
        xlabel   = "Importance Value",
        out_path = PATH_CHART_POLL,
    )


def write_placeholder_charts(model_name: str) -> None:
    """
    Write a uniform warning image to all three chart paths.

    Called when the model family does not support intrinsic importance so
    that dashboards and reports always find image files at the expected paths.

    Parameters
    ----------
    model_name : displayed on the placeholder image
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.axis("off")
    ax.text(
        0.5, 0.62,
        f"{model_name}",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=16, color=_WARN, fontweight="bold",
    )
    ax.text(
        0.5, 0.38,
        "Intrinsic feature importance unavailable.\n"
        "SHAP support planned for Version 2.",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=12, color=_FG,
    )
    plt.tight_layout()
    for path in (PATH_CHART_ALL, PATH_CHART_TOP10, PATH_CHART_POLL):
        plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=_BG)
        logger.info("Placeholder chart written → %s", path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 9. DATA-FILE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def write_importance_csv(imp_df: pd.DataFrame) -> None:
    """
    Persist the full importance DataFrame to a CSV file.

    Parameters
    ----------
    imp_df : importance DataFrame (all features, ranked descending)
    """
    imp_df.to_csv(PATH_CSV, index=False)
    logger.info("Feature importance CSV saved → %s", PATH_CSV)


def write_importance_json(
    imp_df:     pd.DataFrame,
    model_name: str,
    family:     str,
    method:     str,
) -> None:
    """
    Persist the importance DataFrame as structured JSON.

    JSON schema
    ───────────
    {
      "model_name"  : str,
      "family"      : str,
      "method"      : str,
      "timestamp"   : ISO-8601 str,
      "n_features"  : int,
      "importances" : [{rank, feature, importance, importance_pct, …}, …]
    }

    Parameters
    ----------
    imp_df     : importance DataFrame
    model_name : best model name
    family     : detected model family
    method     : human-readable method description
    """
    payload: Dict[str, Any] = {
        "model_name":  model_name,
        "family":      family,
        "method":      method,
        "timestamp":   datetime.now().isoformat(),
        "n_features":  len(imp_df),
        "importances": imp_df.to_dict(orient="records"),
    }
    with open(PATH_JSON_IMP, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Feature importance JSON saved → %s", PATH_JSON_IMP)


def write_empty_data_files(model_name: str, method: str) -> None:
    """
    Write structurally valid but empty data files for unsupported models.

    Ensures downstream consumers (dashboard, reporting) always find the
    expected files and do not raise FileNotFoundError.

    Parameters
    ----------
    model_name : best model name
    method     : importance method label (will indicate unavailability)
    """
    empty_df = pd.DataFrame(
        columns=["rank", "feature", "importance", "importance_pct", "cumulative_pct"]
    )
    empty_df.to_csv(PATH_CSV, index=False)
    logger.info("Empty importance CSV written → %s", PATH_CSV)

    with open(PATH_JSON_IMP, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model_name":  model_name,
                "method":      method,
                "timestamp":   datetime.now().isoformat(),
                "n_features":  0,
                "importances": [],
            },
            fh,
            indent=2,
        )
    logger.info("Empty importance JSON written → %s", PATH_JSON_IMP)


def write_dashboard_json(
    imp_df:     Optional[pd.DataFrame],
    model_name: str,
    family:     str,
    lb_record:  Dict[str, Any],
) -> None:
    """
    Write the compact dashboard JSON consumed by the AirSense AI frontend.

    When feature importance is available the JSON carries the top feature's
    name, raw importance value, importance percentage, the top-10 list, and
    an auto-generated human-readable summary sentence.

    When unavailable the fields are set to null so the dashboard can render
    a graceful "not available" state rather than crashing.

    Dashboard JSON schema
    ─────────────────────
    {
      "model"             : str,
      "family"            : str,
      "validation_r2"     : float | null,
      "engineering_score" : float | null,
      "deployment_status" : str   | null,
      "top_feature"       : str   | null,
      "importance"        : float | null,
      "importance_pct"    : float | null,
      "top10_features"    : [str] | [],
      "summary"           : str,
      "timestamp"         : ISO-8601 str,
      "charts"            : {all_features, top10, pollutants}
    }

    Parameters
    ----------
    imp_df     : importance DataFrame or None
    model_name : best model name
    family     : detected model family
    lb_record  : raw leaderboard record for the best model
    """
    top_feature: Optional[str]   = None
    importance:  Optional[float] = None
    imp_pct:     Optional[float] = None
    top10:       List[str]       = []

    if imp_df is not None and not imp_df.empty:
        row         = imp_df.iloc[0]
        top_feature = str(row["feature"])
        importance  = round(float(row["importance"]),    6)
        imp_pct     = round(float(row["importance_pct"]), 4)
        top10       = imp_df.head(10)["feature"].tolist()
        summary     = (
            f"{top_feature} is the strongest AQI driver in {model_name}, "
            f"accounting for {imp_pct:.1f}% of total feature importance."
        )
    else:
        summary = (
            f"Intrinsic feature importance is not available for {model_name}. "
            "SHAP support is planned for Version 2."
        )

    payload: Dict[str, Any] = {
        "model":             model_name,
        "family":            family,
        "validation_r2":     lb_record.get("validation_r2"),
        "engineering_score": lb_record.get("engineering_score"),
        "deployment_status": lb_record.get("deployment_status"),
        "top_feature":       top_feature,
        "importance":        importance,
        "importance_pct":    imp_pct,
        "top10_features":    top10,
        "summary":           summary,
        "timestamp":         datetime.now().isoformat(),
        "charts": {
            "all_features": PATH_CHART_ALL,
            "top10":        PATH_CHART_TOP10,
            "pollutants":   PATH_CHART_POLL,
        },
    }

    with open(PATH_JSON_DASH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Dashboard JSON saved → %s", PATH_JSON_DASH)


# ─────────────────────────────────────────────────────────────────────────────
# 10. TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_importance_row(
    rank:    int,
    feature: str,
    pct:     float,
    cumul:   float,
) -> str:
    """Format a single table row for the importance section of the report."""
    return f"  {rank:>4}  {feature:<40}  {pct:>9.3f}%  {cumul:>8.3f}%"


def compose_report(
    model_name:  str,
    family:      str,
    method:      str,
    imp_df:      Optional[pd.DataFrame],
    poll_df:     Optional[pd.DataFrame],
    lb_record:   Dict[str, Any],
    elapsed_s:   float,
) -> str:
    """
    Compose the full model explainability text report.

    Sections
    ────────
    1.  Script overview
    2.  Model performance (from leaderboard)
    3.  Importance method
    4.  Top-20 features table
    5.  Least important features (bottom 5)
    6.  Pollutant analysis
    7.  Interpretation
    8.  Recommendation
    9.  Output file paths
    10. Generation time

    Parameters
    ----------
    model_name : best model name
    family     : ``"tree"`` | ``"linear"`` | ``"unsupported"``
    method     : human-readable importance method string
    imp_df     : full importance DataFrame, or None
    poll_df    : pollutant-filtered DataFrame, or None
    lb_record  : raw leaderboard record
    elapsed_s  : pipeline wall-clock time in seconds

    Returns
    -------
    str
    """
    SEP  = "=" * 70
    SEP2 = "-" * 70
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    n_features = len(imp_df) if imp_df is not None else "N/A"

    lines: List[str] = [
        SEP,
        "  AIRSENSE AI – MODEL EXPLAINABILITY REPORT",
        f"  Generated         : {ts}",
        "  Script            : utils/06_model_explainability.py",
        SEP,
        "",
        "[1] OVERVIEW",
        SEP2,
        f"  Best Model        : {model_name}",
        f"  Model Family      : {family}",
        f"  Importance Method : {method}",
        f"  Feature Count     : {n_features}",
        "",
        "[2] MODEL PERFORMANCE  (from leaderboard)",
        SEP2,
        f"  Engineering Score : {lb_record.get('engineering_score', 'N/A')}",
        f"  Validation R²     : {lb_record.get('validation_r2',     'N/A')}",
        f"  Test R²           : {lb_record.get('test_r2',           'N/A')}",
        f"  Validation RMSE   : {lb_record.get('rmse',              'N/A')}",
        f"  Validation MAE    : {lb_record.get('mae',               'N/A')}",
        f"  Deployment Status : {lb_record.get('deployment_status', 'N/A')}",
        "",
        "[3] IMPORTANCE METHOD",
        SEP2,
    ]

    if family == "tree":
        lines += [
            f"  {model_name} exposes .feature_importances_ (Gini impurity / gain).",
            "  Each value is the mean weighted impurity reduction from that feature",
            "  across all decision trees.  Values sum to 1.0.",
        ]
    elif family == "linear":
        lines += [
            f"  {model_name} exposes .coef_ (regression coefficients).",
            "  Absolute coefficient values are used for importance ranking.",
            "  Signed coefficients are preserved in the CSV for directional",
            "  interpretation: positive → increases predicted AQI;",
            "                  negative → decreases predicted AQI.",
        ]
    else:
        lines += [
            f"  {model_name} does not expose intrinsic feature importances.",
            "  ⚠  Intrinsic feature importance unavailable.",
            "     SHAP support planned for Version 2.",
        ]

    # ── Section 4: Top features ───────────────────────────────────────────────
    lines += ["", "[4] TOP FEATURES", SEP2]

    if imp_df is not None and not imp_df.empty:
        top_n = min(20, len(imp_df))
        lines.append(
            f"  {'Rank':>4}  {'Feature':<40}  {'Importance%':>10}  {'Cumulative%':>11}"
        )
        lines.append(
            f"  {'────':>4}  {'───────':─<40}  {'──────────':>10}  {'───────────':>11}"
        )
        for _, row in imp_df.head(top_n).iterrows():
            lines.append(
                _fmt_importance_row(
                    int(row["rank"]),
                    str(row["feature"]),
                    float(row["importance_pct"]),
                    float(row["cumulative_pct"]),
                )
            )

        for threshold in (80.0, 95.0):
            n_needed = (imp_df["cumulative_pct"] <= threshold).sum() + 1
            n_needed = min(n_needed, len(imp_df))
            lines.append(
                f"\n  Features required to explain {threshold:.0f}% "
                f"of total importance: {n_needed}"
            )
    else:
        lines.append("  Feature importance not available for this model.")

    # ── Section 5: Least important features ───────────────────────────────────
    lines += ["", "[5] LEAST IMPORTANT FEATURES  (bottom 5)", SEP2]

    if imp_df is not None and len(imp_df) >= 5:
        for _, row in imp_df.tail(5).iterrows():
            lines.append(
                f"  {int(row['rank']):>4}  "
                f"{str(row['feature']):<40}  "
                f"{float(row['importance_pct']):>9.3f}%"
            )
        lines.append(
            "\n  Features with < 0.1% importance are candidates for removal "
            "in a future feature-selection step."
        )
    else:
        lines.append("  Not available.")

    # ── Section 6: Pollutant analysis ─────────────────────────────────────────
    lines += ["", "[6] POLLUTANT FEATURE ANALYSIS", SEP2]

    if poll_df is not None and not poll_df.empty:
        poll_total = poll_df["importance_pct"].sum()
        lines.append(
            f"  {len(poll_df)} pollutant-related feature(s) account for "
            f"{poll_total:.2f}% of total importance."
        )
        lines.append("")
        lines.append(
            f"  {'Rank':>4}  {'Pollutant Feature':<40}  {'Importance%':>10}"
        )
        lines.append(
            f"  {'────':>4}  {'─────────────────':─<40}  {'──────────':>10}"
        )
        for _, row in poll_df.iterrows():
            lines.append(
                f"  {int(row['rank']):>4}  "
                f"{str(row['feature']):<40}  "
                f"{float(row['importance_pct']):>9.3f}%"
            )
    else:
        lines.append("  No pollutant features found or importance not available.")

    # ── Section 7: Interpretation ──────────────────────────────────────────────
    lines += ["", "[7] INTERPRETATION", SEP2]

    if imp_df is not None and not imp_df.empty:
        top_feat = imp_df.iloc[0]["feature"]
        top_pct  = imp_df.iloc[0]["importance_pct"]

        lines += [
            f"  • '{top_feat}' is the single most influential predictor "
            f"({top_pct:.1f}% of total importance).",
        ]

        if poll_df is not None and not poll_df.empty:
            lines.append(
                f"  • Pollutant features collectively explain "
                f"{poll_df['importance_pct'].sum():.1f}% of model decisions."
            )

        if family == "linear" and "coefficient" in imp_df.columns:
            n_pos = int((imp_df["coefficient"] > 0).sum())
            n_neg = int((imp_df["coefficient"] < 0).sum())
            lines += [
                f"  • {n_pos} feature(s) positively affect predicted AQI.",
                f"  • {n_neg} feature(s) negatively affect predicted AQI.",
            ]

        n_80 = min((imp_df["cumulative_pct"] <= 80.0).sum() + 1, len(imp_df))
        lines.append(
            f"  • {n_80} feature(s) explain 80% of the model's decisions — "
            "a compact, interpretable core."
        )
    else:
        lines += [
            "  Intrinsic feature importance is unavailable for this model family.",
            "  Consider switching to a tree-based model (Random Forest, XGBoost)",
            "  for transparency, or enable SHAP analysis in Version 2.",
        ]

    # ── Section 8: Recommendation ─────────────────────────────────────────────
    lines += ["", "[8] RECOMMENDATION", SEP2]

    status = lb_record.get("deployment_status", "Unknown")

    if family in ("tree", "linear"):
        lines += [
            f"  {model_name} is marked '{status}' by the leaderboard.",
            "",
            "  Suggested next steps:",
            "  • Use feature_importance.csv to identify and remove near-zero",
            "    importance features, then retrain a leaner model.",
            "  • Share the top-10 features chart with domain experts to confirm",
            "    that the model relies on physically meaningful pollutant signals.",
            "  • Verify that PM2.5 / PM10 appear in the top contributors;",
            "    if not, investigate data quality or pipeline configuration.",
        ]
    else:
        lines += [
            f"  {model_name} does not support intrinsic explainability.",
            "  Recommended alternatives:",
            "  • Replace with Random Forest or XGBoost for built-in importances.",
            "  • Enable SHAP post-hoc explainability in Version 2.",
        ]

    # ── Section 9: Output files ────────────────────────────────────────────────
    lines += [
        "",
        "[9] OUTPUT FILES",
        SEP2,
        f"  Feature importance CSV    : {PATH_CSV}",
        f"  Feature importance JSON   : {PATH_JSON_IMP}",
        f"  Dashboard JSON            : {PATH_JSON_DASH}",
        f"  Chart — all features      : {PATH_CHART_ALL}",
        f"  Chart — top 10            : {PATH_CHART_TOP10}",
        f"  Chart — pollutants        : {PATH_CHART_POLL}",
        f"  This report               : {PATH_REPORT}",
        "",
        "[10] GENERATION TIME",
        SEP2,
        f"  Elapsed : {elapsed_s:.3f} seconds",
        "",
        SEP,
        "  Explainability analysis complete.",
        SEP,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Execute the focused model explainability pipeline.

    Steps
    ─────
    1.  Create output directories.
    2.  Read leaderboard and identify the best model.
    3.  Load the best model artefact.
    4.  Build the feature column list from train.csv.
    5.  Detect the model family.
    6.  Extract feature importance.
    7.  Filter pollutant features.
    8.  Generate the three charts (or placeholder charts for unsupported).
    9.  Write CSV, importance JSON, and dashboard JSON.
    10. Compose and save the text report.
    """
    logger.info("=" * 70)
    logger.info("  AirSense AI — Model Explainability Pipeline START")
    logger.info("=" * 70)

    t0 = time.perf_counter()

    # ── 1. Directories ────────────────────────────────────────────────────────
    ensure_output_dirs()

    # ── 2. Leaderboard ────────────────────────────────────────────────────────
    logger.info("Reading leaderboard …")
    leaderboard = read_leaderboard(PATH_LEADERBOARD)
    lb_record   = get_best_model_record(leaderboard)
    model_name  = lb_record["model"]

    # ── 3. Model ──────────────────────────────────────────────────────────────
    logger.info("Loading model '%s' …", model_name)
    model = load_model(model_name, DIR_MODELS)

    # ── 4. Feature columns ────────────────────────────────────────────────────
    logger.info("Reading feature columns from training CSV …")
    feature_cols = read_feature_columns(PATH_TRAIN_CSV)

    # ── 5. Family detection ───────────────────────────────────────────────────
    family = detect_family(model)
    logger.info("Model family: %s", family)

    # Method label used in reports and JSON files
    method_map = {
        "tree":        "Tree Feature Importance (feature_importances_)",
        "linear":      "Linear Coefficient Importance (|coef_|)",
        "unsupported": (
            "Intrinsic importance unavailable — "
            "SHAP support planned for Version 2"
        ),
    }
    method = method_map[family]

    # ── 6. Importance extraction ──────────────────────────────────────────────
    logger.info("Extracting feature importance …")
    imp_df = extract_importance(model, family, feature_cols, model_name)

    # ── 7. Pollutant filter ───────────────────────────────────────────────────
    poll_df: Optional[pd.DataFrame] = None
    if imp_df is not None:
        poll_df = filter_pollutant_rows(imp_df, POLLUTANT_KEYWORDS)
        logger.info("Pollutant features identified: %d", len(poll_df))

    # ── 8. Charts ─────────────────────────────────────────────────────────────
    logger.info("Generating charts …")
    if imp_df is not None:
        plot_all_features(imp_df, model_name, family)
        plot_top10(imp_df, model_name, family)
        plot_pollutants(
            poll_df if poll_df is not None else pd.DataFrame(),
            model_name,
            family,
        )
    else:
        write_placeholder_charts(model_name)

    # ── 9. Data files ─────────────────────────────────────────────────────────
    logger.info("Writing data files …")
    if imp_df is not None:
        write_importance_csv(imp_df)
        write_importance_json(imp_df, model_name, family, method)
    else:
        write_empty_data_files(model_name, method)

    write_dashboard_json(imp_df, model_name, family, lb_record)

    # ── 10. Text report ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    logger.info("Composing explainability report …")

    report_text = compose_report(
        model_name = model_name,
        family     = family,
        method     = method,
        imp_df     = imp_df,
        poll_df    = poll_df,
        lb_record  = lb_record,
        elapsed_s  = elapsed,
    )

    with open(PATH_REPORT, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    logger.info("Explainability report saved → %s", PATH_REPORT)

    print()
    print(report_text)
    print()

    logger.info("=" * 70)
    logger.info(
        "  AirSense AI — Model Explainability Pipeline COMPLETE (%.3f s)",
        elapsed,
    )
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point with structured top-level error handling."""
    try:
        run_pipeline()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    except ValueError as exc:
        logger.error("Data error: %s", exc)
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        raise SystemExit(1) from exc
    except PermissionError as exc:
        logger.error("Permission denied: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()