"""
AirSense AI – Urban Air Quality Intelligence Platform
=====================================================
Module  : utils/prediction_engine.py
Purpose : Production prediction backend for AQI forecasting.

          Orchestrates the full inference pipeline:
            1.  Reads the leaderboard to identify the Production Ready model.
            2.  Loads that model, the KNNImputer, and the StandardScaler.
            3.  Delegates feature construction to feature_builder.py.
            4.  Applies imputation → scaling → model prediction.
            5.  Classifies the predicted AQI using CPCB breakpoints.
            6.  Generates a health advisory matched to the AQI category.
            7.  Writes predictions/latest_prediction.json.
            8.  Appends a row to predictions/prediction_history.csv.
            9.  Returns a single result dictionary to the caller.

          Design constraints
          ──────────────────
          • No terminal interaction (no input(), no print()).
          • No feature engineering logic — always delegates to feature_builder.
          • No model training, plotting, or explainability.
          • Model name is never hardcoded; always read from the leaderboard.

          Integration contract
          ────────────────────
          Caller supplies a raw-input dict and receives a result dict:

              from utils.prediction_engine import predict
              result = predict(raw_inputs)

          See predict() for the full parameter specification.

Author  : AirSense AI Engineering Team
Python  : 3.11+
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# feature_builder lives in the same utils/ package.
# Import at module level so a missing file is caught immediately on import.
try:
    from utils.feature_builder import build_complete_feature_vector
except ModuleNotFoundError:
    # Allow direct execution from the project root without package installation.
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.feature_builder import build_complete_feature_vector  # type: ignore

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PATH_LEADERBOARD        = os.path.join("leaderboard", "leaderboard.csv")
PATH_IMPUTER            = os.path.join("artifacts",   "imputer.joblib")
PATH_SCALER             = os.path.join("artifacts",   "scaler.joblib")
DIR_MODELS              = "models"
DIR_PREDICTIONS         = "predictions"
PATH_LATEST_PREDICTION  = os.path.join(DIR_PREDICTIONS, "latest_prediction.json")
PATH_PREDICTION_HISTORY = os.path.join(DIR_PREDICTIONS, "prediction_history.csv")

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# CPCB AQI category breakpoints (upper-bound inclusive).
# Source: Central Pollution Control Board of India, AQI technical document.
_AQI_BREAKPOINTS: List[tuple[int, str]] = [
    (50,  "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
]

# Health advisories keyed by AQI category.
# Written for a general Indian urban population per CPCB and WHO guidance.
_HEALTH_ADVISORIES: Dict[str, List[str]] = {
    "Good": [
        "Air quality is satisfactory. Enjoy outdoor activities.",
        "Maintain a healthy lifestyle with regular outdoor exercise.",
        "No precautions necessary for the general population.",
    ],
    "Satisfactory": [
        "Air quality is acceptable for most people.",
        "Unusually sensitive individuals should consider limiting prolonged "
        "outdoor exertion.",
        "General population may continue normal activities.",
    ],
    "Moderate": [
        "Sensitive groups (children, elderly, those with respiratory or "
        "cardiac conditions) should reduce prolonged outdoor exertion.",
        "Monitor AQI before outdoor exercise.",
        "Consider wearing a mask if you experience discomfort outdoors.",
        "Keep windows partially closed during peak traffic hours.",
    ],
    "Poor": [
        "Everyone should reduce prolonged or heavy outdoor exertion.",
        "Sensitive groups should avoid all outdoor activities.",
        "Wear an N95 mask if outdoor exposure is unavoidable.",
        "Use air purifiers indoors if available.",
        "Keep windows and doors closed.",
    ],
    "Very Poor": [
        "Avoid all outdoor exercise and prolonged outdoor activity.",
        "Sensitive groups should remain indoors.",
        "Use an air purifier with a HEPA filter indoors.",
        "Wear an N95 or equivalent mask if you must go outside.",
        "Consult a doctor if you experience respiratory or cardiac symptoms.",
    ],
    "Severe": [
        "Stay indoors at all times — air quality poses a serious health risk.",
        "Keep all windows and doors tightly closed.",
        "Wear an N95 or higher-rated mask if outdoor exposure is unavoidable.",
        "Use an air purifier continuously.",
        "Seek medical attention immediately if you experience difficulty "
        "breathing, chest pain, or other severe symptoms.",
        "Authorities may issue emergency advisories — follow official guidance.",
    ],
}

# Columns written to the prediction history CSV (in order).
_HISTORY_COLUMNS: List[str] = [
    "timestamp", "city", "predicted_aqi", "aqi_category",
    "model_used", "validation_r2", "engineering_score",
]

# Columns excluded from the numeric feature matrix before imputation/scaling.
# Must mirror DROP_COLS in 04_train_models.py.
_DROP_COLS: List[str] = [
    "City", "Date", "AQI_Bucket", "Season",
    "temperature", "humidity", "wind_speed", "rainfall",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. LEADERBOARD — IDENTIFY PRODUCTION MODEL
# ─────────────────────────────────────────────────────────────────────────────

def _load_leaderboard(path: str) -> pd.DataFrame:
    """
    Load and validate the leaderboard CSV produced by 05_model_comparison.py.

    Parameters
    ----------
    path : str

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError — file absent
    ValueError        — file empty or missing required columns
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Leaderboard not found at '{path}'. "
            "Run utils/05_model_comparison.py first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Leaderboard at '{path}' is empty. "
            "Re-run utils/05_model_comparison.py."
        )

    required = {"model", "deployment_status"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Leaderboard is missing columns: {missing}. "
            "Re-run utils/05_model_comparison.py."
        )

    return df


def _select_production_model(leaderboard: pd.DataFrame) -> Dict[str, Any]:
    """
    Return the leaderboard record whose deployment_status is 'Production Ready'.

    When multiple models are Production Ready the one with the highest
    engineering_score (or, failing that, rank 1) is selected.

    Parameters
    ----------
    leaderboard : pd.DataFrame

    Returns
    -------
    dict — one leaderboard row as a Python dict

    Raises
    ------
    ValueError — no Production Ready model found
    """
    prod = leaderboard[
        leaderboard["deployment_status"].str.strip() == "Production Ready"
    ].copy()

    if prod.empty:
        raise ValueError(
            "No model with deployment_status='Production Ready' found in the "
            "leaderboard.  Check the output of utils/05_model_comparison.py or "
            "lower the production-readiness thresholds."
        )

    # Prefer highest engineering score; fall back to natural row order
    if "engineering_score" in prod.columns:
        prod = prod.sort_values("engineering_score", ascending=False)
    elif "rank" in prod.columns:
        prod = prod.sort_values("rank", ascending=True)

    record = prod.iloc[0].to_dict()
    logger.info(
        "Production model selected: '%s'  "
        "(Engineering Score=%.4f | Validation R²=%.4f)",
        record["model"],
        float(record.get("engineering_score", float("nan"))),
        float(record.get("validation_r2",     float("nan"))),
    )
    return record


# ─────────────────────────────────────────────────────────────────────────────
# 2. ARTEFACT LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_joblib(path: str, label: str) -> Any:
    """
    Deserialise a joblib artefact with clear, actionable error messages.

    Parameters
    ----------
    path  : str — file path
    label : str — human-readable name for error / log messages

    Returns
    -------
    Any — deserialised object

    Raises
    ------
    FileNotFoundError — file absent
    RuntimeError      — joblib cannot deserialise the file
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label} not found at '{path}'. "
            "Run the appropriate upstream pipeline step."
        )
    try:
        obj = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {label} from '{path}': {exc}"
        ) from exc

    logger.info("%s loaded from: %s", label, path)
    return obj


def _load_model(model_name: str, models_dir: str) -> Any:
    """
    Load the production model artefact.

    Parameters
    ----------
    model_name  : str — leaderboard 'model' value (used as filename stem)
    models_dir  : str — directory containing .joblib files

    Returns
    -------
    fitted sklearn-compatible estimator
    """
    path = os.path.join(models_dir, f"{model_name}.joblib")
    return _load_joblib(path, label=f"Model '{model_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE MATRIX CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_dataframe(
    raw_inputs:        Dict[str, Any],
    training_columns:  Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Call feature_builder and return a one-row DataFrame.

    Feature engineering is performed entirely inside feature_builder.py to
    guarantee training-inference parity.  This function never duplicates
    any feature logic.

    The columns of the returned DataFrame are restricted to the numeric
    model-input columns (DROP_COLS removed, target column removed) and
    ordered to match the training dataset when *training_columns* is provided.

    Parameters
    ----------
    raw_inputs       : dict containing all keys expected by
                       feature_builder.build_complete_feature_vector()
    training_columns : optional ordered list of feature column names as
                       seen by the model during training.  When supplied,
                       the DataFrame is reindexed to exactly that order.

    Returns
    -------
    pd.DataFrame — one row, numeric features only

    Raises
    ------
    KeyError   — a required raw input key is missing
    ValueError — feature_builder validation fails
    """
    feature_dict = build_complete_feature_vector(**raw_inputs)
    df_full      = pd.DataFrame([feature_dict])

    # Drop non-numeric / identifier / placeholder columns
    drop = [c for c in _DROP_COLS if c in df_full.columns]
    df   = df_full.drop(columns=drop)

    # Keep only numeric columns (safety net for any stray object fields)
    df = df.select_dtypes(include=[np.number])

    if training_columns is not None:
        # Reindex to the exact column order used during training.
        # Columns absent from the feature vector receive NaN (handled by imputer).
        df = df.reindex(columns=training_columns)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. IMPUTATION + SCALING
# ─────────────────────────────────────────────────────────────────────────────

def _apply_imputer(
    imputer: Any,
    df:      pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the fitted KNNImputer to *df* and return an imputed DataFrame.

    The imputer transforms a numpy array; we reconstruct the DataFrame
    with the original column list to preserve feature names for the scaler.

    Parameters
    ----------
    imputer : fitted KNNImputer (sklearn)
    df      : feature DataFrame with potential NaN values

    Returns
    -------
    pd.DataFrame — same shape, NaN values filled
    """
    imputed_array = imputer.transform(df)
    return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)


def _apply_scaler(
    scaler,
    df,
):
    
    """
    Scale only the columns that the scaler was trained on.
    Leave all other columns unchanged.
    """

    scale_cols = list(scaler.feature_names_in_)

    scaled = df.copy()

    scaled_values = scaler.transform(df[scale_cols])

    scaled.loc[:, scale_cols] = scaled_values

    return scaled


# ─────────────────────────────────────────────────────────────────────────────
# 5. AQI CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_aqi(aqi: int) -> str:
    """
    Map an integer AQI value to a CPCB category string.

    Breakpoints (upper bound inclusive)
    ────────────────────────────────────
    0 – 50    Good
    51 – 100  Satisfactory
    101 – 200 Moderate
    201 – 300 Poor
    301 – 400 Very Poor
    401 – 500 Severe
    > 500     Severe  (clipped)

    Parameters
    ----------
    aqi : int

    Returns
    -------
    str — one of the six CPCB category labels
    """
    for upper_bound, category in _AQI_BREAKPOINTS:
        if aqi <= upper_bound:
            return category
    return "Severe"   # anything above 500 remains Severe


# ─────────────────────────────────────────────────────────────────────────────
# 6. HEALTH ADVISORY
# ─────────────────────────────────────────────────────────────────────────────

def get_health_advisory(category: str) -> List[str]:
    """
    Return the health advisory list for a given AQI *category*.

    Parameters
    ----------
    category : str — CPCB category label (e.g. "Moderate")

    Returns
    -------
    list[str] — ordered advisory recommendations

    Raises
    ------
    KeyError — when *category* is not one of the six CPCB categories
    """
    if category not in _HEALTH_ADVISORIES:
        raise KeyError(
            f"Unknown AQI category '{category}'. "
            f"Expected one of: {list(_HEALTH_ADVISORIES.keys())}."
        )
    return _HEALTH_ADVISORIES[category]


# ─────────────────────────────────────────────────────────────────────────────
# 7. PERSISTENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _write_latest_prediction(payload: Dict[str, Any]) -> None:
    """
    Overwrite predictions/latest_prediction.json with *payload*.

    Parameters
    ----------
    payload : dict — the complete prediction result
    """
    os.makedirs(DIR_PREDICTIONS, exist_ok=True)
    with open(PATH_LATEST_PREDICTION, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Latest prediction JSON saved → %s", PATH_LATEST_PREDICTION)


def _append_prediction_history(record: Dict[str, Any]) -> None:
    """
    Append one row to predictions/prediction_history.csv.

    The CSV is created with a header on first write.  Subsequent calls
    append without repeating the header so the history is never overwritten.

    Parameters
    ----------
    record : dict — subset of the prediction result matching _HISTORY_COLUMNS
    """
    os.makedirs(DIR_PREDICTIONS, exist_ok=True)
    file_exists = os.path.isfile(PATH_PREDICTION_HISTORY)

    with open(PATH_PREDICTION_HISTORY, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=_HISTORY_COLUMNS,
            extrasaction="ignore",   # ignore keys not in _HISTORY_COLUMNS
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: record.get(col, "") for col in _HISTORY_COLUMNS})

    logger.info("Prediction appended to history → %s", PATH_PREDICTION_HISTORY)


# ─────────────────────────────────────────────────────────────────────────────
# 8. INTERNAL INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class _PredictionEngine:
    """
    Stateful wrapper that holds loaded artefacts across multiple predictions.

    Loading joblib artefacts is expensive.  This class is instantiated once
    (lazily, via the module-level ``_ENGINE`` singleton) and reused for every
    subsequent call to ``predict()``.

    Attributes
    ----------
    model            : fitted production model
    imputer          : fitted KNNImputer
    scaler           : fitted StandardScaler
    model_name       : str — name of the production model
    validation_r2    : float — validation R² from the leaderboard
    engineering_score: float — engineering score from the leaderboard
    training_columns : list[str] | None — feature column order from training
    """

    def __init__(self) -> None:
        self.model:              Any               = None
        self.imputer:            Any               = None
        self.scaler:             Any               = None
        self.model_name:         str               = ""
        self.validation_r2:      float             = float("nan")
        self.engineering_score:  float             = float("nan")
        self.training_columns:   Optional[List[str]] = None
        self._initialised:       bool              = False

    def initialise(self) -> None:
        """
        Load all artefacts from disk.  Called once on first prediction.

        Raises
        ------
        FileNotFoundError — any required file is missing
        RuntimeError      — joblib cannot deserialise a file
        ValueError        — leaderboard has no Production Ready model
        """
        logger.info("Prediction Engine START")

        # ── Leaderboard ───────────────────────────────────────────────────────
        leaderboard = _load_leaderboard(PATH_LEADERBOARD)
        lb_record   = _select_production_model(leaderboard)

        self.model_name       = str(lb_record["model"])
        self.validation_r2    = float(lb_record.get("validation_r2",     float("nan")))
        self.engineering_score = float(lb_record.get("engineering_score", float("nan")))

        # ── Model ─────────────────────────────────────────────────────────────
        logger.info("Loading production model '%s' …", self.model_name)
        self.model = _load_model(self.model_name, DIR_MODELS)

        # Attempt to recover the training column order from the model itself
        # (sklearn ≥ 1.0 stores feature_names_in_ when fit on a DataFrame).
        if hasattr(self.model, "feature_names_in_"):
            self.training_columns = list(self.model.feature_names_in_)
            logger.info(
                "Training column order recovered from model "
                "(%d features).", len(self.training_columns)
            )

        # ── Imputer ───────────────────────────────────────────────────────────
        logger.info("Loading imputer …")
        self.imputer = _load_joblib(PATH_IMPUTER, label="KNNImputer")

        # ── Scaler ────────────────────────────────────────────────────────────
        logger.info("Loading scaler …")
        self.scaler = _load_joblib(PATH_SCALER, label="StandardScaler")

        self._initialised = True
        logger.info("All artefacts loaded successfully.")

    def run(self, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full inference pipeline for one prediction request.

        Parameters
        ----------
        raw_inputs : dict — see predict() for the expected keys

        Returns
        -------
        dict — prediction result (see predict() for the schema)

        Raises
        ------
        ValueError — feature building or AQI classification fails
        RuntimeError — model / imputer / scaler raises during prediction
        """
        if not self._initialised:
            self.initialise()

        ts = datetime.now().isoformat()

        # ── Feature vector ────────────────────────────────────────────────────
        logger.info("Building features …")
        df_features = _build_feature_dataframe(
            raw_inputs       = raw_inputs,
            training_columns = self.training_columns,
        )

        # ── Imputation ────────────────────────────────────────────────────────
        logger.info("Applying imputation …")
        df_imputed = _apply_imputer(self.imputer, df_features)

       # ── Scaling ───────────────────────────────────────────────────────────
        

        logger.info("Applying scaling …")
        scaled_df = _apply_scaler(self.scaler, df_imputed)

        # ── Prediction ────────────────────────────────────────────────────────
        logger.info("Predicting AQI …")
        try:
            raw_pred = self.model.predict(scaled_df)
        except Exception as exc:
            raise RuntimeError(
                f"Model '{self.model_name}' raised an error during prediction: {exc}"
            ) from exc

        predicted_aqi = int(round(float(raw_pred[0])))
        predicted_aqi = max(0, min(500, predicted_aqi))   # clip to valid CPCB range

        # ── Classification ────────────────────────────────────────────────────
        category = classify_aqi(predicted_aqi)
        logger.info(
            "Predicted AQI: %d  |  Category: %s", predicted_aqi, category
        )

        # ── Health advisory ───────────────────────────────────────────────────
        logger.info("Generating health advisory …")
        advisory = get_health_advisory(category)

        city = str(raw_inputs.get("city", "Unknown"))

        # ── Build result dict ─────────────────────────────────────────────────
        result: Dict[str, Any] = {
            "timestamp":        ts,
            "city":             city,
            "predicted_aqi":    predicted_aqi,
            "aqi_category":     category,
            "model_used":       self.model_name,
            "validation_r2":    self.validation_r2,
            "engineering_score": self.engineering_score,
            "health_advisory":  advisory,
        }

        # ── Persist ───────────────────────────────────────────────────────────
        logger.info("Saving latest prediction JSON …")
        _write_latest_prediction(result)

        logger.info("Saving prediction history …")
        _append_prediction_history(result)

        logger.info("Prediction Complete.")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

# One engine instance is shared across all calls to predict() within a process.
# This avoids reloading large .joblib artefacts on every call.
_ENGINE: _PredictionEngine = _PredictionEngine()


# ─────────────────────────────────────────────────────────────────────────────
# 9. PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def predict(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the AirSense AI AQI prediction pipeline for one inference request.

    This is the sole public entry point of this module.  All artefacts are
    loaded lazily on the first call and cached for subsequent calls.

    Parameters
    ----------
    raw_inputs : dict
        All keys required by feature_builder.build_complete_feature_vector().
        Required keys:

        city            : str
        date            : str (YYYY-MM-DD) | datetime.date | datetime.datetime
        pollutants      : dict[str, float | None]
            Mandatory sub-keys: PM2.5, PM10, NO, NO2, NOx, NH3,
                                CO, SO2, O3, Benzene, Toluene
        weather         : dict[str, float | None]
            Optional sub-keys: temperature, humidity, wind_speed, rainfall
        aqi_history     : list[float | None]  — 7 values, oldest first
        pm25_yesterday  : float | None
        pm25_3days_ago  : float | None
        pm10_yesterday  : float | None
        co_yesterday    : float | None

    Returns
    -------
    dict with keys:
        timestamp        : str  — ISO-8601 prediction timestamp
        city             : str
        predicted_aqi    : int  — CPCB AQI (clipped to [0, 500])
        aqi_category     : str  — one of the six CPCB category labels
        model_used       : str  — production model name from leaderboard
        validation_r2    : float
        engineering_score: float
        health_advisory  : list[str] — ordered advisory recommendations

    Raises
    ------
    FileNotFoundError — leaderboard, model, imputer, or scaler missing
    ValueError        — invalid raw inputs or leaderboard format
    RuntimeError      — artefact loading or model inference fails
    KeyError          — unknown AQI category (should never occur in practice)

    Example
    -------
    >>> from utils.prediction_engine import predict
    >>> result = predict({
    ...     "city": "Delhi",
    ...     "date": "2024-03-15",
    ...     "pollutants": {
    ...         "PM2.5": 145.2, "PM10": 210.0, "NO": 18.3,
    ...         "NO2": 42.1, "NOx": 60.4, "NH3": 12.5,
    ...         "CO": 1.8, "SO2": 9.7, "O3": 31.2,
    ...         "Benzene": 2.1, "Toluene": 5.3,
    ...     },
    ...     "weather": {
    ...         "temperature": 28.5, "humidity": 62.0,
    ...         "wind_speed": 3.2, "rainfall": 0.0,
    ...     },
    ...     "aqi_history": [155, 148, 162, 170, 141, 138, 150],
    ...     "pm25_yesterday": 140.1,
    ...     "pm25_3days_ago": 132.5,
    ...     "pm10_yesterday": 200.3,
    ...     "co_yesterday": 1.6,
    ... })
    >>> result["predicted_aqi"]
    163
    >>> result["aqi_category"]
    'Moderate'
    """
    return _ENGINE.run(raw_inputs)


def reload_engine() -> None:
    """
    Force the singleton engine to reload all artefacts on the next call.

    Useful when models, imputers, or scalers have been updated on disk
    without restarting the Python process (e.g. after retraining).
    """
    global _ENGINE
    _ENGINE = _PredictionEngine()
    logger.info(
        "Prediction engine reset — artefacts will be reloaded on next predict() call."
    )