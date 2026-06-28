"""
AirSense AI – Urban Air Quality Intelligence Platform
=====================================================
Script  : utils/07_live_prediction.py
Purpose : Terminal interface for real-time AQI prediction.

          Collects raw user inputs from the terminal, performs lightweight
          formatting validation, assembles the raw-input dictionary, and
          delegates the entire ML pipeline to prediction_engine.predict().

          This script contains NO machine-learning logic.  Feature
          engineering, imputation, scaling, model inference, health advisory
          generation, JSON persistence, and history logging are all handled
          inside prediction_engine.py.

          Responsibilities of this script
          ────────────────────────────────
          ✓  Display a styled terminal banner
          ✓  Prompt the user for every required input
          ✓  Validate formatting (non-empty, numeric, date, non-negative)
          ✓  Build the raw-input dict
          ✓  Call prediction_engine.predict(raw_inputs)
          ✓  Display the prediction result in a formatted summary panel

          Out of scope
          ────────────
          ✗  Feature engineering
          ✗  Loading models / scalers / imputers
          ✗  Generating health advisories
          ✗  Writing JSON or CSV files

Author  : AirSense AI Engineering Team
Python  : 3.11+
"""

import logging
import os
import sys
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Make sure the project root is on sys.path so utils imports resolve ─────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from utils.prediction_engine import predict
except ModuleNotFoundError:
    # Fallback: script is executed from inside utils/ directly
    sys.path.insert(0, os.path.dirname(__file__))
    from prediction_engine import predict  # type: ignore

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_WIDTH = 54   # total width of the terminal panel border

_BANNER = f"""
{'=' * _WIDTH}
{'AirSense AI'.center(_WIDTH)}
{'Real-Time AQI Prediction System'.center(_WIDTH)}
{'=' * _WIDTH}
"""

_SECTION_LINE = "-" * _WIDTH


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    """Print a section heading separator."""
    print(f"\n{_SECTION_LINE}")
    print(f"  {title}")
    print(_SECTION_LINE)


def _print_result_row(label: str, value: Any, indent: int = 2) -> None:
    """Print a single label-value row inside the result panel."""
    pad = " " * indent
    print(f"{pad}{label:<22}: {value}")


def _error(message: str) -> None:
    """Print a user-friendly error message to stderr."""
    print(f"\n  ⚠  {message}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _prompt(
    label:    str,
    required: bool = True,
    default:  Optional[str] = None,
) -> str:
    """
    Prompt the user for a string value with an optional default.

    Parameters
    ----------
    label    : displayed prompt label
    required : when True and the user enters nothing, re-prompt
    default  : pre-filled value shown in brackets; accepted on empty entry

    Returns
    -------
    str — stripped, non-empty (or empty when not required)
    """
    hint = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {label}{hint}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        if not required:
            return ""
        _error(f"'{label}' is required — please enter a value.")


def _prompt_float(
    label:      str,
    required:   bool = True,
    allow_none: bool = False,
) -> Optional[float]:
    """
    Prompt the user for a non-negative numeric value.

    Parameters
    ----------
    label      : displayed prompt label
    required   : re-prompt when True and input is empty
    allow_none : when True, an empty entry returns None instead of re-prompting

    Returns
    -------
    float or None (only when allow_none=True and input is empty)

    Raises
    ------
    KeyboardInterrupt — propagated from input()
    """
    while True:
        raw = input(f"  {label}: ").strip()

        if not raw:
            if allow_none:
                return None
            if not required:
                return None
            _error(f"'{label}' is required — please enter a numeric value.")
            continue

        try:
            value = float(raw)
        except ValueError:
            _error(f"'{raw}' is not a valid number. Please enter a numeric value.")
            continue

        if value < 0:
            _error(
                f"'{label}' cannot be negative ({value}). "
                "Pollutant concentrations must be ≥ 0."
            )
            continue

        return value


def _prompt_date(label: str) -> str:
    """
    Prompt the user for a date in YYYY-MM-DD format.

    Empty input defaults to today's date.

    Parameters
    ----------
    label : displayed prompt label

    Returns
    -------
    str — ISO-8601 date string (YYYY-MM-DD)
    """
    today = date.today().isoformat()
    while True:
        raw = input(f"  {label} [{today}]: ").strip()
        if not raw:
            return today
        try:
            datetime.strptime(raw, "%Y-%m-%d")
            return raw
        except ValueError:
            _error(
                f"'{raw}' is not a valid date. "
                "Please use YYYY-MM-DD format (e.g. 2024-03-15)."
            )


def _prompt_aqi_history() -> List[Optional[float]]:
    """
    Collect the last 7 AQI readings (oldest first, newest last).

    Each value may be left blank to indicate a missing reading.

    Returns
    -------
    list[float | None] of exactly 7 elements
    """
    print("\n  Enter the last 7 AQI values (oldest first).")
    print("  Press Enter to mark a day as missing (None).")

    history: List[Optional[float]] = []
    labels = [
        "AQI 7 days ago",
        "AQI 6 days ago",
        "AQI 5 days ago",
        "AQI 4 days ago",
        "AQI 3 days ago",
        "AQI 2 days ago",
        "AQI yesterday  ",
    ]

    for day_label in labels:
        while True:
            raw = input(f"    {day_label}: ").strip()
            if not raw:
                history.append(None)
                break
            try:
                value = float(raw)
            except ValueError:
                _error(f"'{raw}' is not a valid number.")
                continue
            if value < 0:
                _error(f"AQI cannot be negative ({value}).")
                continue
            if value > 1000:
                _error(f"AQI {value} exceeds the CPCB ceiling of 1000.")
                continue
            history.append(value)
            break

    return history


# ─────────────────────────────────────────────────────────────────────────────
# INPUT COLLECTION SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _collect_city_and_date() -> Tuple[str, str]:
    """
    Collect city name and prediction date from the user.

    Returns
    -------
    (city, date_str)
    """
    _print_section("Location & Date")
    city     = _prompt("City")
    date_str = _prompt_date("Prediction Date")
    return city, date_str


def _collect_pollutants() -> Dict[str, Optional[float]]:
    """
    Collect current-day pollutant concentrations from the user.

    All values are required; None is accepted when a sensor is unavailable.

    Returns
    -------
    dict[str, float | None]
    """
    _print_section("Current Pollutants  (µg/m³ unless noted)")

    pollutant_labels: List[Tuple[str, str]] = [
        ("PM2.5",   "PM2.5  (µg/m³)"),
        ("PM10",    "PM10   (µg/m³)"),
        ("NO",      "NO     (µg/m³)"),
        ("NO2",     "NO2    (µg/m³)"),
        ("NOx",     "NOx    (µg/m³)"),
        ("NH3",     "NH3    (µg/m³)"),
        ("CO",      "CO     (mg/m³)"),
        ("SO2",     "SO2    (µg/m³)"),
        ("O3",      "O3     (µg/m³)"),
        ("Benzene", "Benzene (µg/m³)"),
        ("Toluene", "Toluene (µg/m³)"),
    ]

    pollutants: Dict[str, Optional[float]] = {}
    for key, label in pollutant_labels:
        pollutants[key] = _prompt_float(label, required=True, allow_none=True)

    return pollutants


def _collect_weather() -> Dict[str, Optional[float]]:
    """
    Collect current meteorological readings from the user.

    All weather values are optional; blank entry → None.

    Returns
    -------
    dict[str, float | None]
    """
    _print_section("Weather Conditions  (optional)")
    print("  Press Enter to skip any field.")

    weather_labels: List[Tuple[str, str]] = [
        ("temperature", "Temperature   (°C)"),
        ("humidity",    "Humidity      (%)"),
        ("wind_speed",  "Wind Speed    (m/s)"),
        ("rainfall",    "Rainfall      (mm)"),
    ]

    weather: Dict[str, Optional[float]] = {}
    for key, label in weather_labels:
        weather[key] = _prompt_float(label, required=False, allow_none=True)

    return weather


def _collect_historical() -> Dict[str, Any]:
    """
    Collect historical AQI and pollutant readings from the user.

    Returns
    -------
    dict with keys:
        aqi_history, pm25_yesterday, pm25_3days_ago,
        pm10_yesterday, co_yesterday
    """
    _print_section("Historical Data")

    aqi_history    = _prompt_aqi_history()

    print()
    pm25_yesterday  = _prompt_float(
        "PM2.5 yesterday  (µg/m³)", required=False, allow_none=True
    )
    pm25_3days_ago  = _prompt_float(
        "PM2.5 3 days ago (µg/m³)", required=False, allow_none=True
    )
    pm10_yesterday  = _prompt_float(
        "PM10  yesterday  (µg/m³)", required=False, allow_none=True
    )
    co_yesterday    = _prompt_float(
        "CO    yesterday  (mg/m³)", required=False, allow_none=True
    )

    return {
        "aqi_history":    aqi_history,
        "pm25_yesterday": pm25_yesterday,
        "pm25_3days_ago": pm25_3days_ago,
        "pm10_yesterday": pm10_yesterday,
        "co_yesterday":   co_yesterday,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INPUT COLLECTION ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def collect_inputs() -> Dict[str, Any]:
    """
    Collect and assemble all user inputs into one raw-input dictionary.

    Delegates to section-level helpers; performs no ML or feature logic.

    Returns
    -------
    dict ready to pass directly to prediction_engine.predict()
    """
    logger.info("Collecting inputs …")

    city, date_str = _collect_city_and_date()
    pollutants     = _collect_pollutants()
    weather        = _collect_weather()
    historical     = _collect_historical()

    return {
        "city":       city,
        "date":       date_str,
        "pollutants": pollutants,
        "weather":    weather,
        **historical,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def display_result(result: Dict[str, Any]) -> None:
    """
    Display the prediction result in a formatted terminal panel.

    Parameters
    ----------
    result : dict — the dict returned by prediction_engine.predict()
    """
    print(f"\n{'=' * _WIDTH}")
    print("  Prediction Complete".center(_WIDTH))
    print(f"{'=' * _WIDTH}")

    _print_result_row("City",              result.get("city",             "N/A"))
    _print_result_row("Predicted AQI",     result.get("predicted_aqi",    "N/A"))
    _print_result_row("AQI Category",      result.get("aqi_category",     "N/A"))
    _print_result_row("Model Used",        result.get("model_used",       "N/A"))

    val_r2 = result.get("validation_r2")
    _print_result_row(
        "Validation R²",
        f"{val_r2:.4f}" if val_r2 is not None else "N/A",
    )

    eng_score = result.get("engineering_score")
    _print_result_row(
        "Engineering Score",
        f"{eng_score:.4f}" if eng_score is not None else "N/A",
    )

    # ── Health advisory ───────────────────────────────────────────────────────
    advisory: List[str] = result.get("health_advisory", [])
    if advisory:
        print(f"\n  {'Health Advisory':<22}:")
        for tip in advisory:
            # Wrap long lines at _WIDTH - 6 characters
            words, line = tip.split(), ""
            for word in words:
                if len(line) + len(word) + 1 > _WIDTH - 6:
                    print(f"    • {line.strip()}")
                    line = word + " "
                else:
                    line += word + " "
            if line.strip():
                print(f"    • {line.strip()}" if not line.startswith("•") else f"    {line.strip()}")

    # ── Timestamp ──────────────────────────────────────────────────────────────
    ts = result.get("timestamp", "")
    if ts:
        try:
            ts_fmt = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            ts_fmt = ts
        _print_result_row("\nPrediction Time", ts_fmt)

    print(f"\n{'=' * _WIDTH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point for the AirSense AI terminal prediction interface.

    Flow
    ────
    1.  Display the banner.
    2.  Collect all user inputs.
    3.  Call prediction_engine.predict() with the assembled dict.
    4.  Display the formatted result.

    Exits gracefully on KeyboardInterrupt, ValueError, and unexpected errors.
    """
    logger.info("Live Prediction START")
    print(_BANNER)

    try:
        # ── Collect ───────────────────────────────────────────────────────────
        raw_inputs = collect_inputs()

        # ── Predict ───────────────────────────────────────────────────────────
        logger.info("Calling prediction engine …")
        print(f"\n{'=' * _WIDTH}")
        print("  Running prediction, please wait …")
        print(f"{'=' * _WIDTH}")

        result = predict(raw_inputs)

        # ── Display ───────────────────────────────────────────────────────────
        logger.info("Displaying results …")
        display_result(result)

        logger.info("Live Prediction Complete.")

    except KeyboardInterrupt:
        print(f"\n\n{'=' * _WIDTH}")
        print("  Prediction cancelled by user.".center(_WIDTH))
        print(f"{'=' * _WIDTH}\n")
        logger.info("Live Prediction cancelled by user (KeyboardInterrupt).")
        sys.exit(0)

    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _error(str(exc))
        logger.error("Live Prediction failed: %s", exc)
        sys.exit(1)

    except Exception as exc:  # noqa: BLE001
        _error(f"Unexpected error: {exc}")
        logger.exception("Unexpected error during live prediction.")
        sys.exit(1)


if __name__ == "__main__":
    main()