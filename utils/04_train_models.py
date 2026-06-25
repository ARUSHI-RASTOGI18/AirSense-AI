"""
AirSense AI – Urban Air Quality Intelligence Platform
=====================================================
Script  : utils/04_train_models.py
Purpose : Train all candidate regression models on the prepared AQI dataset,
          record per-model metrics, save trained artefacts, predictions,
          feature importances, and per-model metadata JSON files.

          This script is intentionally limited to TRAINING only.
          Model comparison, leaderboard generation, and winner selection are
          implemented in utils/05_model_evaluation.py.

          Models trained
          ──────────────
          Linear family  : Linear Regression, Ridge, Lasso
                           → trained on SCALED data (gradient-based / distance
                             sensitive; unscaled features with vastly different
                             magnitudes distort the loss surface and prevent
                             coefficient regularisation from working correctly)

          Tree family    : Decision Tree, Random Forest, Extra Trees
                           → trained on UNSCALED data (split-based models are
                             invariant to monotonic feature transformations;
                             scaling neither helps nor hurts them, and keeping
                             raw values makes importances easier to interpret)

          Boosting family: XGBoost, LightGBM, CatBoost  (optional — skipped
                           gracefully if the library is not installed)
                           → trained on UNSCALED data (same reasoning as trees)

          Neural network : MLP Regressor
                           → trained on SCALED data (gradient descent diverges
                             or converges extremely slowly when input features
                             span different orders of magnitude; standardisation
                             to zero-mean unit-variance is standard practice)

          Key design decisions
          ────────────────────
          1. Scale routing happens automatically inside the training loop via
             a MODEL_USES_SCALED_DATA set — no code duplication.

          2. random_state=42 is applied wherever the API allows it.

          3. Metrics are evaluated on all three splits (train, validation, test)
             so that 05_model_evaluation.py can detect overfitting without
             reloading artefacts.

          4. Prediction CSVs now include Actual_AQI, Predicted_AQI, Residual,
             Absolute_Error, Split, and Index for dashboard / residual analysis.

          5. Linear model importance files export coefficient, absolute_
             coefficient instead of the tree-style feature_importances_ value.

          6. A metadata JSON is saved to models_metadata/ for every trained
             model containing name, timestamp, hyperparameters, data shape,
             and timing — for deployment and dashboard consumption.

          7. MAPE is computed with an epsilon guard to avoid division-by-zero.

Author  : AirSense AI Engineering Team
"""

import json
import logging
import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# ── Optional boosting libraries ───────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress sklearn convergence warnings so project log stays clean
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── File paths ────────────────────────────────────────────────────────────────
# Unscaled splits — used by tree / boosting models
PATH_TRAIN       = os.path.join("datasets", "prepared", "train.csv")
PATH_VALID       = os.path.join("datasets", "prepared", "validation.csv")
PATH_TEST        = os.path.join("datasets", "prepared", "test.csv")

# Scaled splits — used by linear / neural models
PATH_TRAIN_SC    = os.path.join("datasets", "prepared", "train_scaled.csv")
PATH_VALID_SC    = os.path.join("datasets", "prepared", "validation_scaled.csv")
PATH_TEST_SC     = os.path.join("datasets", "prepared", "test_scaled.csv")

DIR_MODELS        = "models"
DIR_METADATA      = "models_metadata"
DIR_PREDICTIONS   = "predictions"
DIR_METRICS       = "metrics"
DIR_REPORTS       = "reports"
DIR_IMPORTANCES   = os.path.join("reports", "feature_importances")

PATH_TRAINING_SUMMARY = os.path.join(DIR_REPORTS, "training_summary.txt")

# ── Modelling constants ───────────────────────────────────────────────────────
TARGET_COL   = "AQI"
RANDOM_STATE = 42

# Columns removed from the feature matrix before training.
# Weather placeholders remain 100% NaN (populated later by a weather API
# fetch) and cannot be used by sklearn models.  We drop them universally
# so all models receive an identical feature schema.
DROP_COLS: List[str] = [
    "City", "Date", "AQI_Bucket", "Season",
    "temperature", "humidity", "wind_speed", "rainfall",
]

# Models that require standardised (zero-mean, unit-variance) inputs.
# WHY linear models need scaling:
#   Ridge / Lasso penalise the L2 / L1 norm of the coefficient vector.
#   If features have very different magnitudes (CO ~ 0-5, PM2.5 ~ 0-500),
#   the regulariser shrinks large-scale coefficients more aggressively,
#   biasing the model.  StandardScaler places all features on equal footing
#   so the regulariser operates fairly.
# WHY MLP needs scaling:
#   Gradient descent updates are proportional to the input magnitude.
#   Unscaled inputs cause slow convergence on low-magnitude features and
#   numerical instability on high-magnitude ones.
MODEL_USES_SCALED_DATA: Set[str] = {
    "LinearRegression",
    "Ridge",
    "Lasso",
    "MLP",
}

# Epsilon guard for MAPE denominator (avoids ZeroDivisionError when AQI == 0)
MAPE_EPSILON = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# 1. I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for directory in [
        DIR_MODELS, DIR_METADATA, DIR_PREDICTIONS,
        DIR_METRICS, DIR_REPORTS, DIR_IMPORTANCES,
    ]:
        os.makedirs(directory, exist_ok=True)
    logger.info("Output directories verified / created.")


def load_split(path: str, label: str) -> pd.DataFrame:
    """
    Load a prepared CSV split and parse the Date column.

    Parameters
    ----------
    path  : str
    label : str — used only in log messages

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
    pd.errors.EmptyDataError
    """
    logger.info("Loading %s from: %s", label, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[{label}] file not found: '{path}'. "
            "Run utils/03_data_preparation.py first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    if df.empty:
        raise pd.errors.EmptyDataError(f"[{label}] file is empty: '{path}'")
    logger.info("%s loaded — %d rows × %d cols", label, *df.shape)
    return df


def build_feature_matrix(
    df:         pd.DataFrame,
    drop_cols:  List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate the feature matrix X from the target series y.

    Columns in *drop_cols* and *target_col* are excluded from X.
    A secondary select_dtypes guard ensures no stray object columns
    reach the estimators.

    Parameters
    ----------
    df         : pd.DataFrame
    drop_cols  : non-feature columns to exclude
    target_col : regression target column name

    Returns
    -------
    X : pd.DataFrame  — numeric feature matrix
    y : pd.Series     — target vector
    """
    to_drop = [c for c in drop_cols + [target_col] if c in df.columns]
    X = df.drop(columns=to_drop).select_dtypes(include=[np.number])
    y = df[target_col]
    return X, y


def save_model(model: Any, name: str) -> str:
    """
    Persist a trained model via joblib.

    Returns
    -------
    str — path where the model was saved
    """
    path = os.path.join(DIR_MODELS, f"{name}.joblib")
    joblib.dump(model, path)
    logger.info("[%s] Model saved → %s", name, path)
    return path


def save_predictions(
    name:        str,
    split_data:  List[Tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """
    Save enriched prediction CSV for all splits.

    Each row contains:
        Split           — 'train' | 'validation' | 'test'
        Index           — row index within that split
        Actual_AQI      — ground-truth AQI value
        Predicted_AQI   — model prediction
        Residual        — Actual_AQI − Predicted_AQI
        Absolute_Error  — |Residual|

    These columns enable residual analysis and dashboard visualisations
    in later pipeline stages without needing to reload the raw datasets.

    Parameters
    ----------
    name       : model name (used as filename stem)
    split_data : list of (split_label, y_true_array, y_pred_array)
    """
    frames = []
    for split_label, y_true, y_pred in split_data:
        residual = y_true - y_pred
        frames.append(pd.DataFrame({
            "Split":          split_label,
            "Index":          np.arange(len(y_true)),
            "Actual_AQI":     y_true,
            "Predicted_AQI":  y_pred,
            "Residual":       residual,
            "Absolute_Error": np.abs(residual),
        }))

    out  = pd.concat(frames, ignore_index=True)
    path = os.path.join(DIR_PREDICTIONS, f"{name}_predictions.csv")
    out.to_csv(path, index=False)
    logger.info("[%s] Predictions saved → %s", name, path)


def save_metrics(metrics: Dict[str, Any], name: str) -> None:
    """Persist the metrics dictionary as a JSON file."""
    path = os.path.join(DIR_METRICS, f"{name}_metrics.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("[%s] Metrics saved → %s", name, path)


def save_metadata(
    name:          str,
    model:         Any,
    n_features:    int,
    train_rows:    int,
    train_time:    float,
    pred_time:     float,
    uses_scaled:   bool,
) -> None:
    """
    Save a per-model metadata JSON to models_metadata/.

    Contents
    --------
    model_name       : str
    timestamp        : ISO-8601 string
    n_features       : int  — number of features in the input matrix
    training_rows    : int  — rows in the training split used
    hyperparameters  : dict — extracted from model.get_params()
    train_time_s     : float
    prediction_time_s: float
    random_state     : int or None
    uses_scaled_data : bool

    This file is consumed by deployment and dashboard modules to reproduce
    the training configuration without loading the full model artefact.

    Parameters
    ----------
    name        : model identifier (filename stem)
    model       : fitted estimator
    n_features  : number of input features
    train_rows  : number of training rows
    train_time  : wall-clock fit time in seconds
    pred_time   : wall-clock predict time in seconds (validation split)
    uses_scaled : whether this model trained on scaled data
    """
    # get_params() is available on all sklearn-compatible estimators
    try:
        hyperparams = model.get_params()
        # Convert non-serialisable values (e.g. callables) to strings
        hyperparams = {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in hyperparams.items()
        }
    except Exception:
        hyperparams = {}

    metadata = {
        "model_name":         name,
        "timestamp":          datetime.now().isoformat(),
        "n_features":         n_features,
        "training_rows":      train_rows,
        "hyperparameters":    hyperparams,
        "train_time_s":       round(train_time, 6),
        "prediction_time_s":  round(pred_time,  6),
        "random_state":       RANDOM_STATE,
        "uses_scaled_data":   uses_scaled,
    }

    path = os.path.join(DIR_METADATA, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("[%s] Metadata saved → %s", name, path)


def save_feature_importance(
    model:        Any,
    feature_cols: List[str],
    name:         str,
) -> None:
    """
    Export feature importances / coefficients to CSV.

    Tree / boosting models  → exports ``feature_importances_``
                              columns: feature, importance
                              sorted by importance descending

    Linear models           → exports ``coef_``
                              columns: feature, coefficient,
                                       absolute_coefficient
                              sorted by absolute_coefficient descending

    WHY different formats?
    ─────────────────────
    Tree importances are always non-negative (they measure average impurity
    reduction), so a single "importance" column suffices.
    Linear coefficients carry sign (direction of effect) which is meaningful
    — a large negative coefficient for PM2.5 would indicate an inverse
    relationship.  We preserve the raw coefficient and add an absolute
    version for ranking.

    Parameters
    ----------
    model        : fitted estimator
    feature_cols : ordered list of feature names
    name         : model name (used as filename stem)
    """
    path = os.path.join(DIR_IMPORTANCES, f"{name}_feature_importance.csv")

    # ── Tree / boosting models ─────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        if len(imp) != len(feature_cols):
            logger.warning(
                "[%s] feature_importances_ length mismatch (%d vs %d) — skipping.",
                name, len(imp), len(feature_cols),
            )
            return
        df = (
            pd.DataFrame({"feature": feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        df.to_csv(path, index=False)
        logger.info("[%s] Feature importances (tree) saved → %s", name, path)
        return

    # ── Linear models ──────────────────────────────────────────────────────
    if hasattr(model, "coef_"):
        coef = model.coef_
        if len(coef) != len(feature_cols):
            logger.warning(
                "[%s] coef_ length mismatch (%d vs %d) — skipping.",
                name, len(coef), len(feature_cols),
            )
            return
        df = (
            pd.DataFrame({
                "feature":             feature_cols,
                "coefficient":         coef,
                "absolute_coefficient": np.abs(coef),
            })
            .sort_values("absolute_coefficient", ascending=False)
            .reset_index(drop=True)
        )
        df.to_csv(path, index=False)
        logger.info("[%s] Feature importances (linear coef) saved → %s", name, path)
        return

    logger.debug("[%s] No importance/coef attribute found — skipping.", name)


# ─────────────────────────────────────────────────────────────────────────────
# 2. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error with an epsilon denominator guard.

    MAPE = mean( |y_true − y_pred| / max(|y_true|, ε) ) × 100

    Returns
    -------
    float — MAPE as a percentage
    """
    denom = np.maximum(np.abs(y_true), MAPE_EPSILON)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def evaluate_split(
    model: Any,
    X:     pd.DataFrame,
    y:     pd.Series,
    label: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate predictions and compute R², MAE, RMSE, MAPE for one split.

    Parameters
    ----------
    model : fitted estimator
    X     : feature matrix
    y     : true target values
    label : 'train' | 'validation' | 'test'

    Returns
    -------
    y_pred  : np.ndarray
    metrics : dict
    """
    t0     = time.perf_counter()
    y_pred = model.predict(X)
    pred_t = time.perf_counter() - t0

    y_true = y.to_numpy()

    m = {
        f"{label}_r2":        round(float(r2_score(y_true, y_pred)),                   6),
        f"{label}_mae":       round(float(mean_absolute_error(y_true, y_pred)),         6),
        f"{label}_rmse":      round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        f"{label}_mape":      round(compute_mape(y_true, y_pred),                       6),
        f"{label}_pred_time": round(pred_t,                                             6),
    }
    return y_pred, m


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def build_model_registry() -> List[Tuple[str, Any]]:
    """
    Construct an ordered list of (name, unfitted_estimator) pairs.

    Hyperparameter notes
    ────────────────────
    • Random Forest / Extra Trees: n_estimators raised to 300 for better
      variance reduction without a prohibitive runtime cost.
    • XGBoost: n_estimators reduced to 300 (the 500-tree default trained
      longer without meaningful metric improvement on this dataset size).
    • MLP: hidden_layer_sizes changed from (128, 128) to (128, 64) — the
      second layer is narrowed to act as a compression bottleneck, which
      tends to generalise better on tabular regression tasks of this size.

    Returns
    -------
    list of (name, estimator)
    """
    registry: List[Tuple[str, Any]] = []

    # ── Linear family (will use SCALED data) ─────────────────────────────────
    registry.append(("LinearRegression", LinearRegression()))

    registry.append(("Ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)))

    registry.append(("Lasso", Lasso(
        alpha=0.1,
        max_iter=5000,
        random_state=RANDOM_STATE,
    )))

    # ── Tree family (will use UNSCALED data) ──────────────────────────────────
    registry.append(("DecisionTree", DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )))

    registry.append(("RandomForest", RandomForestRegressor(
        n_estimators=300,        # increased from 200 for better variance reduction
        max_depth=None,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )))

    registry.append(("ExtraTrees", ExtraTreesRegressor(
        n_estimators=300,        # increased from 200
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )))

    # ── Boosting family (will use UNSCALED data) ──────────────────────────────
    if _XGBOOST_AVAILABLE:
        registry.append(("XGBoost", XGBRegressor(
            n_estimators=300,    # reduced from 500; same accuracy, faster training
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )))
    else:
        logger.warning(
            "XGBoost not installed — skipping XGBRegressor. "
            "Install with: pip install xgboost"
        )

    if _LIGHTGBM_AVAILABLE:
        registry.append(("LightGBM", LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=63,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )))
    else:
        logger.warning(
            "LightGBM not installed — skipping LGBMRegressor. "
            "Install with: pip install lightgbm"
        )

    if _CATBOOST_AVAILABLE:
        registry.append(("CatBoost", CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        )))
    else:
        logger.warning(
            "CatBoost not installed — skipping CatBoostRegressor. "
            "Install with: pip install catboost"
        )

    # ── Neural network (will use SCALED data) ─────────────────────────────────
    # hidden_layer_sizes changed to (128, 64): the narrower second layer acts
    # as a compression bottleneck which improves generalisation on tabular data.
    registry.append(("MLP", MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )))

    logger.info(
        "Model registry built — %d model(s) queued: %s",
        len(registry),
        [n for n, _ in registry],
    )
    return registry


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA ROUTING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def select_data_for_model(
    name:    str,
    unscaled: Tuple[pd.DataFrame, pd.Series,
                    pd.DataFrame, pd.Series,
                    pd.DataFrame, pd.Series],
    scaled:   Tuple[pd.DataFrame, pd.Series,
                    pd.DataFrame, pd.Series,
                    pd.DataFrame, pd.Series],
) -> Tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    bool,
]:
    """
    Route each model to the correct dataset (scaled vs unscaled).

    WHY automatic routing?
    ─────────────────────
    Manually writing separate training calls for each model group creates
    code duplication and maintenance risk.  A single membership check against
    MODEL_USES_SCALED_DATA keeps the logic in one place and makes it trivial
    to reclassify a model in the future.

    Parameters
    ----------
    name     : model identifier
    unscaled : tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test)
               built from train.csv / validation.csv / test.csv
    scaled   : tuple of same structure built from *_scaled.csv files

    Returns
    -------
    X_train, y_train, X_valid, y_valid, X_test, y_test, uses_scaled
    """
    uses_scaled = name in MODEL_USES_SCALED_DATA

    if uses_scaled:
        X_tr, y_tr, X_va, y_va, X_te, y_te = scaled
        logger.info("[%s] Using SCALED data.", name)
    else:
        X_tr, y_tr, X_va, y_va, X_te, y_te = unscaled
        logger.info("[%s] Using UNSCALED data.", name)

    return X_tr, y_tr, X_va, y_va, X_te, y_te, uses_scaled


# ─────────────────────────────────────────────────────────────────────────────
# 5. SINGLE-MODEL TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_single_model(
    name:         str,
    model:        Any,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    X_valid:      pd.DataFrame,
    y_valid:      pd.Series,
    X_test:       pd.DataFrame,
    y_test:       pd.Series,
    feature_cols: List[str],
    uses_scaled:  bool,
) -> Optional[Dict[str, Any]]:
    """
    Fit one model, record all metrics, and save all artefacts.

    A try/except wraps the entire function so a failure in one model never
    aborts the remaining training loop — the error is logged and None is
    returned to signal failure to the caller.

    Parameters
    ----------
    name         : unique model identifier (filesystem-safe)
    model        : unfitted sklearn-compatible estimator
    X_train, y_train, X_valid, y_valid, X_test, y_test : data for this model
    feature_cols : ordered list of feature column names
    uses_scaled  : whether this model trained on standardised data

    Returns
    -------
    dict — all metrics, or None if training failed
    """
    logger.info("─" * 64)
    logger.info("[%s] Training started …", name)

    try:
        # ── Fit ───────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        logger.info("[%s] Fit complete in %.2f s", name, train_time)

        # ── Evaluate all splits ───────────────────────────────────────────────
        preds_train, m_train = evaluate_split(model, X_train, y_train, "train")
        preds_valid, m_valid = evaluate_split(model, X_valid, y_valid, "validation")
        preds_test,  m_test  = evaluate_split(model, X_test,  y_test,  "test")

        # Capture validation prediction time for metadata
        val_pred_time = m_valid["validation_pred_time"]

        # ── Aggregate metrics ─────────────────────────────────────────────────
        all_metrics: Dict[str, Any] = {
            "model_name":     name,
            "uses_scaled":    uses_scaled,
            "train_time_s":   round(train_time, 4),
            **m_train,
            **m_valid,
            **m_test,
            "timestamp":      datetime.now().isoformat(),
        }

        # ── Console summary ───────────────────────────────────────────────────
        logger.info(
            "[%s]  Train R²=%.4f | Val R²=%.4f | Test R²=%.4f | "
            "Val RMSE=%.2f | Val MAE=%.2f | data=%s",
            name,
            all_metrics["train_r2"],
            all_metrics["validation_r2"],
            all_metrics["test_r2"],
            all_metrics["validation_rmse"],
            all_metrics["validation_mae"],
            "scaled" if uses_scaled else "unscaled",
        )

        # ── Save artefacts ────────────────────────────────────────────────────
        save_model(model, name)

        # Enriched prediction CSV (Actual, Predicted, Residual, AbsError)
        save_predictions(
            name=name,
            split_data=[
                ("train",      y_train.to_numpy(), preds_train),
                ("validation", y_valid.to_numpy(), preds_valid),
                ("test",       y_test.to_numpy(),  preds_test),
            ],
        )

        save_metrics(all_metrics, name)

        save_feature_importance(model, feature_cols, name)

        save_metadata(
            name=name,
            model=model,
            n_features=len(feature_cols),
            train_rows=len(X_train),
            train_time=train_time,
            pred_time=val_pred_time,
            uses_scaled=uses_scaled,
        )

        return all_metrics

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[%s] Training FAILED — %s: %s",
            name, type(exc).__name__, exc,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_training_summary(
    results:        List[Dict[str, Any]],
    skipped:        List[str],
    feature_cols:   List[str],
    train_shape:    Tuple[int, int],
    valid_shape:    Tuple[int, int],
    test_shape:     Tuple[int, int],
    elapsed_total:  float,
) -> str:
    """
    Compose the training summary report as a formatted string.

    Parameters
    ----------
    results       : list of metric dicts (one per successfully trained model)
    skipped       : model names that failed
    feature_cols  : feature column names used during training
    train_shape   : (rows, cols) of unscaled training feature matrix
    valid_shape   : (rows, cols) of unscaled validation feature matrix
    test_shape    : (rows, cols) of unscaled test feature matrix
    elapsed_total : wall-clock time for entire training run

    Returns
    -------
    str
    """
    sep  = "=" * 76
    sep2 = "-" * 76

    lines = [
        sep,
        "  AIRSENSE AI – MODEL TRAINING SUMMARY",
        f"  Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "  Script        : utils/04_train_models.py",
        "  Purpose       : Training metrics only. "
        "Ranking in 05_model_evaluation.py.",
        sep,
        "",
        "[1] DATASET SHAPES (UNSCALED REFERENCE)",
        sep2,
        f"  Train      : {train_shape[0]:>8,} rows × {train_shape[1]} features",
        f"  Validation : {valid_shape[0]:>8,} rows × {valid_shape[1]} features",
        f"  Test       : {test_shape[0]:>8,} rows × {test_shape[1]} features",
        f"  Target     : {TARGET_COL}",
        "",
        "[2] DATA ROUTING",
        sep2,
        "  SCALED data used for   : "
        + ", ".join(sorted(MODEL_USES_SCALED_DATA)),
        "  UNSCALED data used for : tree / boosting models",
        "",
        "[3] FEATURE COLUMNS USED",
        sep2,
        f"  Total features : {len(feature_cols)}",
    ]

    for i, col in enumerate(feature_cols, 1):
        lines.append(f"    {i:>3}. {col}")

    lines += [
        "",
        "[4] TRAINING RESULTS",
        sep2,
        f"  {'Model':<22} {'Data':>8} {'Train R²':>9} {'Val R²':>9} "
        f"{'Test R²':>9} {'Val RMSE':>10} {'Val MAE':>10} {'Train(s)':>10}",
        sep2,
    ]

    for r in results:
        data_tag = "scaled" if r.get("uses_scaled") else "unscal"
        lines.append(
            f"  {r['model_name']:<22} {data_tag:>8} "
            f"{r['train_r2']:>9.4f} "
            f"{r['validation_r2']:>9.4f} "
            f"{r['test_r2']:>9.4f} "
            f"{r['validation_rmse']:>10.2f} "
            f"{r['validation_mae']:>10.2f} "
            f"{r['train_time_s']:>10.2f}"
        )

    if skipped:
        lines += ["", "[5] SKIPPED / FAILED MODELS", sep2]
        for name in skipped:
            lines.append(f"  ✗  {name}")

    lines += [
        "",
        "[6] ARTEFACT LOCATIONS",
        sep2,
        f"  Trained models         : {DIR_MODELS}/",
        f"  Model metadata         : {DIR_METADATA}/",
        f"  Predictions            : {DIR_PREDICTIONS}/",
        f"  Per-model JSON metrics : {DIR_METRICS}/",
        f"  Feature importances    : {DIR_IMPORTANCES}/",
        f"  This report            : {PATH_TRAINING_SUMMARY}",
        "",
        "[7] PROCESSING TIME",
        sep2,
        f"  Total elapsed time     : {elapsed_total:.2f} seconds",
        "",
        sep,
        "  Training complete.  Next stage: utils/05_model_evaluation.py",
        sep,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Execute the full model training pipeline.

    Steps
    -----
    1.  Create output directories
    2.  Load unscaled splits (train / validation / test)
    3.  Load scaled splits  (train_scaled / validation_scaled / test_scaled)
    4.  Build feature matrices for both dataset versions
    5.  Build model registry
    6.  Iterate — route each model to correct data, fit, evaluate, save
    7.  Generate and save training summary report
    """
    logger.info("=" * 76)
    logger.info("  AirSense AI — Model Training Pipeline START")
    logger.info("=" * 76)

    pipeline_start = time.perf_counter()

    # ── Step 1: Directories ───────────────────────────────────────────────────
    ensure_dirs()

    # ── Step 2: Load unscaled splits ──────────────────────────────────────────
    df_train    = load_split(PATH_TRAIN, "Train (unscaled)")
    df_valid    = load_split(PATH_VALID, "Validation (unscaled)")
    df_test     = load_split(PATH_TEST,  "Test (unscaled)")

    # ── Step 3: Load scaled splits ────────────────────────────────────────────
    df_train_sc = load_split(PATH_TRAIN_SC, "Train (scaled)")
    df_valid_sc = load_split(PATH_VALID_SC, "Validation (scaled)")
    df_test_sc  = load_split(PATH_TEST_SC,  "Test (scaled)")

    # ── Step 4: Feature matrices ──────────────────────────────────────────────
    X_train,    y_train    = build_feature_matrix(df_train,    DROP_COLS, TARGET_COL)
    X_valid,    y_valid    = build_feature_matrix(df_valid,    DROP_COLS, TARGET_COL)
    X_test,     y_test     = build_feature_matrix(df_test,     DROP_COLS, TARGET_COL)

    X_train_sc, y_train_sc = build_feature_matrix(df_train_sc, DROP_COLS, TARGET_COL)
    X_valid_sc, y_valid_sc = build_feature_matrix(df_valid_sc, DROP_COLS, TARGET_COL)
    X_test_sc,  y_test_sc  = build_feature_matrix(df_test_sc,  DROP_COLS, TARGET_COL)

    feature_cols:    List[str] = list(X_train.columns)
    feature_cols_sc: List[str] = list(X_train_sc.columns)

    # Sanity check — both versions must share identical feature schemas
    assert feature_cols == feature_cols_sc, (
        "Unscaled and scaled feature columns differ — "
        "re-run utils/03_data_preparation.py."
    )

    # All splits within each version must share the same schema
    for tag, X in [("valid", X_valid), ("test", X_test)]:
        assert list(X.columns) == feature_cols, \
            f"Unscaled {tag} columns differ from train."
    for tag, X in [("valid_sc", X_valid_sc), ("test_sc", X_test_sc)]:
        assert list(X.columns) == feature_cols_sc, \
            f"Scaled {tag} columns differ from train_scaled."

    logger.info(
        "Feature matrix shapes — "
        "Unscaled Train: %s | Valid: %s | Test: %s  |  "
        "Scaled  Train: %s | Valid: %s | Test: %s",
        X_train.shape,    X_valid.shape,    X_test.shape,
        X_train_sc.shape, X_valid_sc.shape, X_test_sc.shape,
    )

    # Bundle for routing helper
    unscaled_bundle = (X_train,    y_train,    X_valid,    y_valid,    X_test,    y_test)
    scaled_bundle   = (X_train_sc, y_train_sc, X_valid_sc, y_valid_sc, X_test_sc, y_test_sc)

    # ── Step 5: Model registry ────────────────────────────────────────────────
    registry = build_model_registry()

    # ── Step 6: Training loop ─────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    skipped: List[str]            = []

    for name, model in registry:
        # Automatically route to scaled or unscaled data
        X_tr, y_tr, X_va, y_va, X_te, y_te, uses_scaled = select_data_for_model(
            name, unscaled_bundle, scaled_bundle
        )

        # Use feature_cols — same for both versions (asserted above)
        outcome = train_single_model(
            name=name,
            model=model,
            X_train=X_tr, y_train=y_tr,
            X_valid=X_va, y_valid=y_va,
            X_test=X_te,  y_test=y_te,
            feature_cols=feature_cols,
            uses_scaled=uses_scaled,
        )

        if outcome is not None:
            results.append(outcome)
        else:
            skipped.append(name)

    logger.info("─" * 76)
    logger.info(
        "Training loop complete — %d succeeded, %d skipped/failed.",
        len(results), len(skipped),
    )

    # ── Step 7: Training summary ──────────────────────────────────────────────
    elapsed_total = time.perf_counter() - pipeline_start

    report_text = build_training_summary(
        results=results,
        skipped=skipped,
        feature_cols=feature_cols,
        train_shape=X_train.shape,
        valid_shape=X_valid.shape,
        test_shape=X_test.shape,
        elapsed_total=elapsed_total,
    )

    print()
    print(report_text)
    print()

    os.makedirs(DIR_REPORTS, exist_ok=True)
    with open(PATH_TRAINING_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    logger.info("Training summary saved → %s", PATH_TRAINING_SUMMARY)

    logger.info("=" * 76)
    logger.info(
        "  AirSense AI — Model Training Pipeline COMPLETE (%.2f s)",
        elapsed_total,
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
    except pd.errors.EmptyDataError as exc:
        logger.error("Dataset is empty: %s", exc)
        raise SystemExit(1) from exc
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        raise SystemExit(1) from exc
    except AssertionError as exc:
        logger.error("Schema mismatch: %s", exc)
        raise SystemExit(1) from exc
    except PermissionError as exc:
        logger.error("Permission denied when writing output: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error during model training: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()