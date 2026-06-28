"""
AirSense AI – Urban Air Quality Intelligence Platform
=====================================================
Module  : utils/feature_builder.py
Purpose : Reusable inference-time Feature Builder that guarantees
          training-inference feature parity with utils/02_feature_engineering.py.

          Every formula, weight, constant, and feature name in this module
          is a direct port of the corresponding logic in 02_feature_engineering.py
          and 01_preprocess.py.  Nothing is invented here.  If either upstream
          script changes, this file must be updated to match.

          Responsibilities
          ────────────────
          ✓  Validate raw pollutant, weather, and historical inputs
          ✓  Build date / calendar features  (Year … DayOfWeek_cos)
          ✓  Compute the weighted pollution index
          ✓  Build AQI and pollutant lag features
          ✓  Build rolling-window AQI statistics
          ✓  Build day-over-day change (delta) features
          ✓  Return one flat feature dictionary ready for pd.DataFrame()

          Out of scope
          ────────────
          ✗  Loading models or scalers
          ✗  Predicting AQI
          ✗  Generating charts, reports, CSV, JSON, or dashboard outputs

          Usage
          ─────
          >>> from utils.feature_builder import build_complete_feature_vector
          >>> import pandas as pd
          >>> fv = build_complete_feature_vector(
          ...     city="Delhi", date="2024-03-15",
          ...     pollutants={"PM2.5": 145.2, "PM10": 210.0, "NO": 18.3,
          ...                 "NO2": 42.1, "NOx": 60.4, "NH3": 12.5,
          ...                 "CO": 1.8, "SO2": 9.7, "O3": 31.2,
          ...                 "Benzene": 2.1, "Toluene": 5.3},
          ...     weather={"temperature": 28.5, "humidity": 62.0,
          ...              "wind_speed": 3.2, "rainfall": 0.0},
          ...     aqi_history=[155, 148, 162, 170, 141, 138, 150],
          ...     pm25_yesterday=140.1, pm25_3days_ago=132.5,
          ...     pm10_yesterday=200.3, co_yesterday=1.6,
          ... )
          >>> row = pd.DataFrame([fv])

Author  : AirSense AI Engineering Team
Python  : 3.11+
"""

import logging
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS
# All constants mirror their counterparts in 01_preprocess.py and
# 02_feature_engineering.py.  Change them only when the training pipeline
# changes.
# ─────────────────────────────────────────────────────────────────────────────

# Pollution index weights — must match POLLUTION_INDEX_WEIGHTS in
# 02_feature_engineering.py exactly.
_POLLUTION_WEIGHTS: Dict[str, float] = {
    "PM2.5": 0.40,   # dominant particulate AQI driver (CPCB)
    "PM10":  0.30,   # coarse particulate — road / construction dust
    "NO2":   0.20,   # secondary photochemical pollutant
    "CO":    0.10,   # combustion / vehicular emission indicator
}

# Training-dataset min/max bounds used for pollution-index normalisation.
# These replicate the dataset-level statistics computed inside
# add_pollution_index() in 02_feature_engineering.py.
# Source: CPCB city_day.csv (2015-2020 full dataset).
# If the training dataset changes, recalculate and update these values.
_TRAIN_BOUNDS: Dict[str, Dict[str, float]] = {
    "PM2.5": {"min": 0.0,  "max": 1000.0},
    "PM10":  {"min": 0.0,  "max": 2000.0},
    "NO2":   {"min": 0.0,  "max": 330.0},
    "CO":    {"min": 0.0,  "max": 175.0},
}

# India seasonal month mapping — must match SEASON_MAP in 01_preprocess.py.
_SEASON_MAP: Dict[int, str] = {
    1: "Winter",       2: "Winter",
    3: "Summer",       4: "Summer",       5: "Summer",
    6: "Monsoon",      7: "Monsoon",      8: "Monsoon",  9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
    12: "Winter",
}

# All pollutant keys the caller must supply (values may be None).
_REQUIRED_POLLUTANT_KEYS: List[str] = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene",
]

# Number of historical AQI readings required for lag / rolling features.
_AQI_HISTORY_LEN: int = 7

# ── Type aliases ───────────────────────────────────────────────────────────────
DateLike    = Union[str, date, datetime]
FeatureDict = Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRIVATE UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(value: DateLike) -> date:
    """
    Coerce *value* to a ``datetime.date`` object.

    Parameters
    ----------
    value : str (YYYY-MM-DD) | datetime.date | datetime.datetime

    Returns
    -------
    datetime.date

    Raises
    ------
    TypeError  — unrecognised input type
    ValueError — string cannot be parsed as ISO-8601
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse date '{value}'. "
                "Expected ISO-8601 format YYYY-MM-DD."
            ) from exc
    raise TypeError(
        f"'date' must be str, datetime.date, or datetime.datetime — "
        f"got {type(value).__name__}."
    )


def _coerce_float(value: Any) -> Optional[float]:
    """
    Return ``float(value)`` or ``None`` for None / NaN inputs.

    Parameters
    ----------
    value : Any

    Returns
    -------
    float | None
    """
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _assert_non_negative(value: float, label: str) -> None:
    """
    Raise ``ValueError`` when *value* is a finite negative number.

    ``NaN`` is accepted because it represents a missing sensor reading
    and is handled by the downstream KNNImputer.

    Parameters
    ----------
    value : float
    label : str — field name used in the error message

    Raises
    ------
    TypeError  — value is not numeric
    ValueError — value is a finite negative number
    """
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"'{label}' must be numeric; got {type(value).__name__}."
        )
    if not math.isnan(float(value)) and float(value) < 0:
        raise ValueError(
            f"'{label}' must be ≥ 0; received {value}. "
            "Pollutant concentrations and AQI values cannot be negative."
        )


def _safe_subtract(
    a: Optional[float],
    b: Optional[float],
) -> Optional[float]:
    """Return a − b, or None when either operand is unavailable."""
    fa, fb = _coerce_float(a), _coerce_float(b)
    if fa is None or fb is None:
        return None
    return fa - fb


def _finite_values(seq: List[Optional[float]]) -> List[float]:
    """Return only the finite (non-None, non-NaN) floats from *seq*."""
    result: List[float] = []
    for v in seq:
        f = _coerce_float(v)
        if f is not None:
            result.append(f)
    return result


def _safe_mean(values: List[float]) -> Optional[float]:
    """Arithmetic mean, or None for an empty list."""
    return sum(values) / len(values) if values else None


def _safe_std(values: List[float]) -> Optional[float]:
    """
    Sample standard deviation (ddof=1), or None when fewer than 2 values.

    Mirrors pandas Series.std(ddof=1) used by the rolling std in
    02_feature_engineering.py.
    """
    n = len(values)
    if n < 2:
        return None
    mu       = sum(values) / n
    variance = sum((x - mu) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def _safe_max(values: List[float]) -> Optional[float]:
    """Max, or None for an empty list."""
    return max(values) if values else None


def _safe_min(values: List[float]) -> Optional[float]:
    """Min, or None for an empty list."""
    return min(values) if values else None


def _round_opt(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    """Round *value* to *ndigits* decimal places, or return None."""
    return round(value, ndigits) if value is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# 2. INPUT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_inputs(
    city:            str,
    date_input:      DateLike,
    pollutants:      Dict[str, Optional[float]],
    aqi_history:     List[Optional[float]],
    pm25_yesterday:  Optional[float],
    pm25_3days_ago:  Optional[float],
    pm10_yesterday:  Optional[float],
    co_yesterday:    Optional[float],
) -> None:
    """
    Validate every raw input before feature engineering begins.

    Checks
    ──────
    • *city* is a non-empty string.
    • *date_input* is parseable as a valid calendar date.
    • *pollutants* contains all required keys; non-None values are ≥ 0.
    • *aqi_history* has exactly ``_AQI_HISTORY_LEN`` elements;
      non-None values are in [0, 1000].
    • Historical pollutant scalars, when provided, are ≥ 0.

    Parameters
    ----------
    city           : city name
    date_input     : forecast date
    pollutants     : current-day pollutant concentrations
    aqi_history    : 7 most recent AQI readings (oldest first, newest last)
    pm25_yesterday : PM2.5 one day prior (may be None)
    pm25_3days_ago : PM2.5 three days prior (may be None)
    pm10_yesterday : PM10 one day prior (may be None)
    co_yesterday   : CO one day prior (may be None)

    Raises
    ------
    TypeError  — wrong Python type for any argument
    ValueError — invalid value (negative, out-of-range, wrong length, …)
    """
    # City
    if not isinstance(city, str) or not city.strip():
        raise ValueError(
            f"'city' must be a non-empty string; got {city!r}."
        )

    # Date
    _parse_date(date_input)   # raises on invalid input

    # Pollutant dict
    if not isinstance(pollutants, dict):
        raise TypeError(
            f"'pollutants' must be a dict; got {type(pollutants).__name__}."
        )

    missing = [k for k in _REQUIRED_POLLUTANT_KEYS if k not in pollutants]
    if missing:
        raise ValueError(
            f"'pollutants' is missing required keys: {missing}. "
            f"All of {_REQUIRED_POLLUTANT_KEYS} must be present "
            "(set the value to None if sensor data is unavailable)."
        )

    for key in _REQUIRED_POLLUTANT_KEYS:
        v = pollutants[key]
        if v is not None:
            _assert_non_negative(float(v), label=f"pollutants['{key}']")

    # AQI history
    if not isinstance(aqi_history, list):
        raise TypeError(
            f"'aqi_history' must be a list; got {type(aqi_history).__name__}."
        )
    if len(aqi_history) != _AQI_HISTORY_LEN:
        raise ValueError(
            f"'aqi_history' must contain exactly {_AQI_HISTORY_LEN} elements "
            f"(oldest first, newest last); received {len(aqi_history)}."
        )
    for i, v in enumerate(aqi_history):
        if v is not None:
            _assert_non_negative(float(v), label=f"aqi_history[{i}]")
            if float(v) > 1000:
                raise ValueError(
                    f"aqi_history[{i}] = {v} exceeds the CPCB AQI ceiling of 1000."
                )

    # Historical pollutant scalars
    historical_scalars = [
        ("pm25_yesterday",  pm25_yesterday),
        ("pm25_3days_ago",  pm25_3days_ago),
        ("pm10_yesterday",  pm10_yesterday),
        ("co_yesterday",    co_yesterday),
    ]
    for label, v in historical_scalars:
        if v is not None:
            _assert_non_negative(float(v), label=label)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATE / CALENDAR FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_date_features(date_input: DateLike) -> FeatureDict:
    """
    Generate all calendar-derived features for a single date.

    Mirrors
    ───────
    • engineer_calendar_features() in utils/01_preprocess.py
    • engineer_season()            in utils/01_preprocess.py
    • add_cyclical_encoding()      in utils/02_feature_engineering.py

    Features produced
    ─────────────────
    Year          : calendar year (int)
    Month         : 1–12 (int)
    Day           : day of month 1–31 (int)
    DayOfWeek     : Monday=0 … Sunday=6 (int)
    IsWeekend     : 1 if Sat/Sun else 0 (int)
    Season        : India season label (str)
    Month_sin     : sin(2π × Month / 12)
    Month_cos     : cos(2π × Month / 12)
    DayOfWeek_sin : sin(2π × DayOfWeek / 7)
    DayOfWeek_cos : cos(2π × DayOfWeek / 7)

    Parameters
    ----------
    date_input : DateLike

    Returns
    -------
    dict[str, int | float | str]

    Raises
    ------
    ValueError — when *date_input* cannot be parsed
    """
    dt  = _parse_date(date_input)
    mon = dt.month
    dow = dt.weekday()   # Monday=0, Sunday=6

    return {
        "Year":          dt.year,
        "Month":         mon,
        "Day":           dt.day,
        "DayOfWeek":     dow,
        "IsWeekend":     int(dow >= 5),
        "Season":        _SEASON_MAP.get(mon, "Unknown"),
        # Cyclical encoding projects periodic integers onto a unit circle so
        # that distance-based and gradient-based models understand wrap-around
        # (e.g. month 12 is adjacent to month 1).
        # Formula mirrors add_cyclical_encoding() in 02_feature_engineering.py.
        "Month_sin":     round(math.sin(2 * math.pi * mon / 12), 10),
        "Month_cos":     round(math.cos(2 * math.pi * mon / 12), 10),
        "DayOfWeek_sin": round(math.sin(2 * math.pi * dow / 7),  10),
        "DayOfWeek_cos": round(math.cos(2 * math.pi * dow / 7),  10),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. POLLUTION INDEX
# ─────────────────────────────────────────────────────────────────────────────

def calculate_pollution_index(
    pollutants: Dict[str, Optional[float]],
) -> float:
    """
    Compute the weighted composite pollution index.

    This is an exact port of add_pollution_index() in
    utils/02_feature_engineering.py.

    Formula
    ───────
    pollution_index = Σ  weight_i × clamp( (value_i − min_i) / (max_i − min_i) )

    where clamp(x) = max(0, min(1, x)) to keep inference values in range.

    Weights  (must match _POLLUTION_WEIGHTS / POLLUTION_INDEX_WEIGHTS)
    ──────────────────────────────────────────────────────────────────
    PM2.5 : 0.40   PM10 : 0.30   NO2 : 0.20   CO : 0.10

    A None or NaN pollutant value contributes 0 to the composite —
    the same as a reading at the training minimum.

    Parameters
    ----------
    pollutants : dict[str, float | None]

    Returns
    -------
    float  — composite score in [0, 1], rounded to 6 d.p.
    """
    composite = 0.0

    for pollutant, weight in _POLLUTION_WEIGHTS.items():
        raw = _coerce_float(pollutants.get(pollutant))
        if raw is None:
            continue   # treat missing as min → normalised = 0, no contribution

        bounds = _TRAIN_BOUNDS[pollutant]
        lo, hi = bounds["min"], bounds["max"]

        if hi == lo:
            normalised = 0.0
        else:
            normalised = (raw - lo) / (hi - lo)
            normalised = max(0.0, min(1.0, normalised))   # clamp to [0, 1]

        composite += weight * normalised

    return round(composite, 6)


# ─────────────────────────────────────────────────────────────────────────────
# 5. LAG FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_lag_features(
    aqi_history:    List[Optional[float]],
    pm25_yesterday: Optional[float],
    pm25_3days_ago: Optional[float],
    pm10_yesterday: Optional[float],
    co_yesterday:   Optional[float],
) -> FeatureDict:
    """
    Build all lag features from historical readings.

    Mirrors the per-city groupby().shift() calls in
    utils/02_feature_engineering.py:
        add_aqi_lag_features(), add_pm25_lag_features(),
        add_pm10_lag_features(), add_co_lag_features().

    At inference time a DataFrame with a full city history is unavailable,
    so historical values are supplied explicitly.

    AQI history index convention (oldest → newest)
    ───────────────────────────────────────────────
    index 0  →  t-7  →  AQI_lag_7
    index 1  →  t-6
    index 2  →  t-5
    index 3  →  t-4
    index 4  →  t-3  →  AQI_lag_3
    index 5  →  t-2  →  AQI_lag_2
    index 6  →  t-1  →  AQI_lag_1

    Parameters
    ----------
    aqi_history    : list of 7 AQI readings, oldest first
    pm25_yesterday : PM2.5 at t-1 (may be None)
    pm25_3days_ago : PM2.5 at t-3 (may be None)
    pm10_yesterday : PM10 at t-1  (may be None)
    co_yesterday   : CO at t-1    (may be None)

    Returns
    -------
    dict[str, float | None]

    Raises
    ------
    ValueError — when aqi_history does not have exactly 7 elements
    """
    if len(aqi_history) != _AQI_HISTORY_LEN:
        raise ValueError(
            f"aqi_history must have exactly {_AQI_HISTORY_LEN} elements; "
            f"received {len(aqi_history)}."
        )

    return {
        "AQI_lag_1":  _coerce_float(aqi_history[6]),
        "AQI_lag_2":  _coerce_float(aqi_history[5]),
        "AQI_lag_3":  _coerce_float(aqi_history[4]),
        "AQI_lag_7":  _coerce_float(aqi_history[0]),
        "PM25_lag_1": _coerce_float(pm25_yesterday),
        "PM25_lag_3": _coerce_float(pm25_3days_ago),
        "PM10_lag_1": _coerce_float(pm10_yesterday),
        "CO_lag_1":   _coerce_float(co_yesterday),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. ROLLING FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_rolling_features(
    aqi_history: List[Optional[float]],
) -> FeatureDict:
    """
    Compute rolling AQI statistics from the last 7 observed values.

    Mirrors add_rolling_features() in utils/02_feature_engineering.py.

    Training logic: each rolling window is applied to the *shifted* series
    (.shift(1) before .rolling()), meaning the rolling features for day t
    are computed from days t-7 … t-1.  At inference time aqi_history
    already contains those 7 past readings, so we compute directly without
    any additional shift.

    Features produced
    ─────────────────
    AQI_roll_mean_3 : mean of the 3 most recent readings  (indices 4, 5, 6)
    AQI_roll_mean_7 : mean of all 7 readings
    AQI_roll_std_7  : sample std (ddof=1) of all 7 readings
    AQI_roll_max_7  : maximum of all 7 readings
    AQI_roll_min_7  : minimum of all 7 readings

    None / NaN values are excluded before computing each statistic, which
    mirrors pandas' default skipna=True behaviour.

    Parameters
    ----------
    aqi_history : list[float | None] of exactly 7 elements (oldest first)

    Returns
    -------
    dict[str, float | None]

    Raises
    ------
    ValueError — when aqi_history does not have exactly 7 elements
    """
    if len(aqi_history) != _AQI_HISTORY_LEN:
        raise ValueError(
            f"aqi_history must have exactly {_AQI_HISTORY_LEN} elements; "
            f"received {len(aqi_history)}."
        )

    all7   = _finite_values(aqi_history)
    last3  = _finite_values(aqi_history[-3:])   # indices 4, 5, 6

    return {
        "AQI_roll_mean_3": _round_opt(_safe_mean(last3)),
        "AQI_roll_mean_7": _round_opt(_safe_mean(all7)),
        "AQI_roll_std_7":  _round_opt(_safe_std(all7)),
        "AQI_roll_max_7":  _round_opt(_safe_max(all7)),
        "AQI_roll_min_7":  _round_opt(_safe_min(all7)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. CHANGE / TREND FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_change_features(
    current_pm25:   Optional[float],
    current_pm10:   Optional[float],
    pm25_yesterday: Optional[float],
    pm10_yesterday: Optional[float],
) -> FeatureDict:
    """
    Compute day-over-day change (delta) features.

    Mirrors add_trend_features() in utils/02_feature_engineering.py.

    Formula: change_t = value_t − value_(t-1)

    A positive value means the pollutant worsened since yesterday;
    negative means it improved.

    AQI_change note
    ───────────────
    During training AQI_change = AQI(t) − AQI(t-1) is computed on the
    historical column.  At inference time AQI(t) is the unknown target,
    so AQI_change cannot be computed and is set to None.  This produces
    the same NaN seen for the very first row of each city group during
    training, and is handled gracefully by the downstream KNNImputer.

    Parameters
    ----------
    current_pm25   : today's PM2.5 reading
    current_pm10   : today's PM10 reading
    pm25_yesterday : PM2.5 at t-1
    pm10_yesterday : PM10 at t-1

    Returns
    -------
    dict[str, float | None]
    """
    return {
        "AQI_change":  None,   # unknowable at inference time — see docstring
        "PM25_change": _coerce_float(_safe_subtract(current_pm25, pm25_yesterday)),
        "PM10_change": _coerce_float(_safe_subtract(current_pm10, pm10_yesterday)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. COMPLETE FEATURE VECTOR
# ─────────────────────────────────────────────────────────────────────────────

def build_complete_feature_vector(
    city:            str,
    date:            DateLike,
    pollutants:      Dict[str, Optional[float]],
    weather:         Dict[str, Optional[float]],
    aqi_history:     List[Optional[float]],
    pm25_yesterday:  Optional[float],
    pm25_3days_ago:  Optional[float],
    pm10_yesterday:  Optional[float],
    co_yesterday:    Optional[float],
) -> FeatureDict:
    """
    Build the complete feature dictionary for one inference sample.

    Orchestrates validation and all sub-functions, then returns a single
    flat dictionary whose keys **exactly** match the column names produced
    by the training pipeline:

        01_preprocess.py  →  02_feature_engineering.py  →  03_data_preparation.py

    The returned dict can be converted directly to a one-row pd.DataFrame()
    and passed to the fitted StandardScaler and then to the production model.

    Parameters
    ----------
    city : str
        City name (e.g. ``"Delhi"``).  Stored for traceability; not used
        in any arithmetic feature.
    date : DateLike
        Forecast date as str (YYYY-MM-DD), datetime.date, or datetime.datetime.
    pollutants : dict[str, float | None]
        Current-day pollutant concentrations.
        Required keys: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3,
                       Benzene, Toluene.
        Set a value to None when the sensor reading is unavailable.
    weather : dict[str, float | None]
        Meteorological readings for today.
        Accepted keys: temperature, humidity, wind_speed, rainfall.
        During training these columns are 100 % NaN (placeholder); real
        values may be supplied here for future model versions.
    aqi_history : list[float | None]
        The 7 most recent AQI observations, **oldest first**
        (index 0 = t-7, index 6 = t-1).
    pm25_yesterday  : float | None — PM2.5 at t-1
    pm25_3days_ago  : float | None — PM2.5 at t-3
    pm10_yesterday  : float | None — PM10 at t-1
    co_yesterday    : float | None — CO at t-1

    Returns
    -------
    dict[str, Any]
        Flat mapping of every feature name → value.
        None-valued entries become NaN when converted to pd.DataFrame.

    Raises
    ------
    TypeError  — wrong Python type for any argument
    ValueError — invalid / out-of-range value
    """
    logger.info("Feature Builder START")

    # ── Validate ──────────────────────────────────────────────────────────────
    logger.info("Validating inputs …")
    validate_inputs(
        city            = city,
        date_input      = date,
        pollutants      = pollutants,
        aqi_history     = aqi_history,
        pm25_yesterday  = pm25_yesterday,
        pm25_3days_ago  = pm25_3days_ago,
        pm10_yesterday  = pm10_yesterday,
        co_yesterday    = co_yesterday,
    )

    # ── Date features ─────────────────────────────────────────────────────────
    logger.info("Generating date features …")
    date_feats = build_date_features(date)

    # ── Lag features ──────────────────────────────────────────────────────────
    logger.info("Generating lag features …")
    lag_feats = build_lag_features(
        aqi_history     = aqi_history,
        pm25_yesterday  = pm25_yesterday,
        pm25_3days_ago  = pm25_3days_ago,
        pm10_yesterday  = pm10_yesterday,
        co_yesterday    = co_yesterday,
    )

    # ── Rolling features ──────────────────────────────────────────────────────
    logger.info("Generating rolling features …")
    roll_feats = build_rolling_features(aqi_history)

    # ── Change features ───────────────────────────────────────────────────────
    logger.info("Generating change features …")
    change_feats = build_change_features(
        current_pm25   = pollutants.get("PM2.5"),
        current_pm10   = pollutants.get("PM10"),
        pm25_yesterday = pm25_yesterday,
        pm10_yesterday = pm10_yesterday,
    )

    # ── Pollution index ───────────────────────────────────────────────────────
    logger.info("Generating pollution index …")
    pollution_idx = calculate_pollution_index(pollutants)

    # ── Assemble ──────────────────────────────────────────────────────────────
    logger.info("Building feature vector …")

    feature_vector: FeatureDict = {
        # Identifiers — kept for traceability; excluded by model via DROP_COLS
        "City":  city,
        "Date":  _parse_date(date).isoformat(),

        # Raw pollutant concentrations
        "PM2.5":   _coerce_float(pollutants.get("PM2.5")),
        "PM10":    _coerce_float(pollutants.get("PM10")),
        "NO":      _coerce_float(pollutants.get("NO")),
        "NO2":     _coerce_float(pollutants.get("NO2")),
        "NOx":     _coerce_float(pollutants.get("NOx")),
        "NH3":     _coerce_float(pollutants.get("NH3")),
        "CO":      _coerce_float(pollutants.get("CO")),
        "SO2":     _coerce_float(pollutants.get("SO2")),
        "O3":      _coerce_float(pollutants.get("O3")),
        "Benzene": _coerce_float(pollutants.get("Benzene")),
        "Toluene": _coerce_float(pollutants.get("Toluene")),

        # Weather placeholders (100 % NaN during training; real values allowed)
        "temperature": _coerce_float(weather.get("temperature")),
        "humidity":    _coerce_float(weather.get("humidity")),
        "wind_speed":  _coerce_float(weather.get("wind_speed")),
        "rainfall":    _coerce_float(weather.get("rainfall")),

        # Calendar + cyclical features
        **date_feats,

        # Weighted pollution composite
        "pollution_index": pollution_idx,

        # Lag features
        **lag_feats,

        # Rolling AQI statistics
        **roll_feats,

        # Day-over-day change features
        **change_feats,
    }

    logger.info(
        "Feature Builder Complete — %d features built for %s on %s.",
        len(feature_vector),
        city,
        _parse_date(date).isoformat(),
    )

    return feature_vector