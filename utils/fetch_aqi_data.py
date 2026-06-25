# ─────────────────────────────────────────────────────────────
#  AirSense AI – AQI Data Fetcher
#  Source: OpenAQ API v3 (free, no key required for basic use)
#  Output: datasets/raw/aqi_data.csv
# ─────────────────────────────────────────────────────────────

import requests
import pandas as pd
from datetime import datetime
import os
import time

# ── Target cities with their OpenAQ location search terms ──
CITIES = {
    "Delhi":     "Delhi",
    "Noida":     "Noida",
    "Mumbai":    "Mumbai",
    "Bengaluru": "Bengaluru",
    "Chennai":   "Chennai",
}

# ── OpenAQ v3 base URL ──────────────────────────────────────
BASE_URL = "https://api.openaq.org/v3/locations"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "AirSenseAI/1.0"
}

# ── AQI breakpoints for PM2.5 (India NAQI standard) ────────
def calculate_aqi(pm25):
    """
    Calculate AQI from PM2.5 concentration using India NAQI breakpoints.
    Returns integer AQI or None if input is invalid.
    """
    if pm25 is None or pm25 < 0:
        return None

    breakpoints = [
        # (C_low, C_high, I_low, I_high)
        (0.0,  30.0,   0,  50),
        (30.1, 60.0,  51, 100),
        (60.1, 90.0, 101, 200),
        (90.1,120.0, 201, 300),
        (120.1,250.0,301, 400),
        (250.1,500.0,401, 500),
    ]

    for (c_lo, c_hi, i_lo, i_hi) in breakpoints:
        if c_lo <= pm25 <= c_hi:
            aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
            return round(aqi)

    return 500  # beyond highest breakpoint → hazardous

def get_aqi_category(aqi):
    """Return human-readable AQI category string."""
    if aqi is None:
        return "Unknown"
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
    return "Severe"

def fetch_city_data(city_name, search_term):
    """
    Fetch latest sensor readings for a city from OpenAQ v3 /locations endpoint.
    Returns a list of dicts, one per sensor reading found.
    """
    records = []

    try:
        # Search for monitoring locations in this city
        params = {
            "city":    search_term,
            "limit":   5,           # top 5 stations per city
            "page":    1,
            "order_by": "lastUpdated",
            "sort":    "desc",
        }

        response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        locations = data.get("results", [])

        if not locations:
            print(f"  [WARN] No locations found for {city_name}")
            return records

        for loc in locations:
            station_name = loc.get("name", "Unknown Station")
            country      = loc.get("country", {}).get("name", "India")
            coordinates  = loc.get("coordinates", {})
            lat          = coordinates.get("latitude")
            lon          = coordinates.get("longitude")
            last_updated = loc.get("datetimeLast", {}).get("local", "N/A")

            # Each location has a list of sensors/parameters
            sensors = loc.get("sensors", [])

            # Collect pollutant values from this station
            pollutants = {}
            for sensor in sensors:
                param_name = sensor.get("parameter", {}).get("name", "").lower()
                latest_val = sensor.get("latest", {})
                value      = latest_val.get("value")

                if param_name and value is not None:
                    pollutants[param_name] = round(float(value), 2)

            # Calculate AQI from PM2.5 if available
            pm25_val = pollutants.get("pm25") or pollutants.get("pm2.5")
            aqi      = calculate_aqi(pm25_val)
            category = get_aqi_category(aqi)

            record = {
                "city":            city_name,
                "station":         station_name,
                "country":         country,
                "latitude":        lat,
                "longitude":       lon,
                "last_updated":    last_updated,
                "pm25":            pollutants.get("pm25") or pollutants.get("pm2.5"),
                "pm10":            pollutants.get("pm10"),
                "no2":             pollutants.get("no2"),
                "so2":             pollutants.get("so2"),
                "co":              pollutants.get("co"),
                "o3":              pollutants.get("o3"),
                "calculated_aqi":  aqi,
                "aqi_category":    category,
                "fetched_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            records.append(record)

    except requests.exceptions.HTTPError as e:
        print(f"  [HTTP ERROR] {city_name}: {e}")
    except requests.exceptions.ConnectionError:
        print(f"  [CONNECTION ERROR] {city_name}: Could not reach OpenAQ API")
    except requests.exceptions.Timeout:
        print(f"  [TIMEOUT] {city_name}: Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"  [REQUEST ERROR] {city_name}: {e}")
    except Exception as e:
        print(f"  [UNEXPECTED ERROR] {city_name}: {e}")

    return records


def main():
    print("=" * 60)
    print("  AirSense AI – AQI Data Fetcher")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_records = []

    for city_name, search_term in CITIES.items():
        print(f"\n[→] Fetching data for: {city_name}")
        records = fetch_city_data(city_name, search_term)

        if records:
            all_records.extend(records)
            print(f"  [✓] {len(records)} station(s) found for {city_name}")
        else:
            # Add a placeholder row so the city always appears in the CSV
            all_records.append({
                "city":           city_name,
                "station":        "No Data",
                "country":        "India",
                "latitude":       None,
                "longitude":      None,
                "last_updated":   "N/A",
                "pm25":           None,
                "pm10":           None,
                "no2":            None,
                "so2":            None,
                "co":             None,
                "o3":             None,
                "calculated_aqi": None,
                "aqi_category":   "Unknown",
                "fetched_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"  [!] Placeholder row added for {city_name}")

        # Polite delay between requests to avoid rate limiting
        time.sleep(1)

    # ── Build DataFrame ─────────────────────────────────────
    df = pd.DataFrame(all_records)

    # Ensure correct column order
    column_order = [
        "city", "station", "country", "latitude", "longitude",
        "last_updated", "pm25", "pm10", "no2", "so2", "co", "o3",
        "calculated_aqi", "aqi_category", "fetched_at"
    ]
    df = df.reindex(columns=column_order)

    # ── Save to CSV ─────────────────────────────────────────
    output_dir  = os.path.join("datasets", "raw")
    output_path = os.path.join(output_dir, "aqi_data.csv")

    os.makedirs(output_dir, exist_ok=True)  # create folder if not exists
    df.to_csv(output_path, index=False)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  [✓] Data saved to: {output_path}")
    print(f"  [✓] Total rows:    {len(df)}")
    print(f"  [✓] Columns:       {list(df.columns)}")
    print("\n  City Summary:")
    print("-" * 60)

    summary = df.groupby("city")["calculated_aqi"].agg(
        Stations="count",
        Avg_AQI="mean",
        Max_AQI="max"
    ).round(1)
    print(summary.to_string())

    print("=" * 60)
    print("  AirSense AI – Fetch Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()