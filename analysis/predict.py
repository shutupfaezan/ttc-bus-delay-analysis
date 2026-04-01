import os
import pickle
import numpy as np
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "random_forest_ttc.pkl")

# ── Exact feature schema from training (35 columns) ───────────────────────────
FEATURE_COLUMNS = [
    "Route_Number", "Hour", "Month", "Is_Rush_Hour", "Is_Weekend",
    "Temp_C", "Visibility_km", "Wind_Spd_kmh", "Rel_Humidity_pct",
    "Day_Friday", "Day_Monday", "Day_Saturday", "Day_Sunday",
    "Day_Thursday", "Day_Tuesday", "Day_Wednesday",
    "Season_Fall", "Season_Spring", "Season_Summer", "Season_Winter",
    "Incident_Category_Cleaning", "Incident_Category_Collision",
    "Incident_Category_Diversion", "Incident_Category_Emergency Services",
    "Incident_Category_General Delay", "Incident_Category_Investigation",
    "Incident_Category_Late Departure", "Incident_Category_Mechanical",
    "Incident_Category_Operator", "Incident_Category_Other",
    "Incident_Category_Passenger", "Incident_Category_Security",
    "Incident_Category_Traffic", "Incident_Category_Utilized Off Route",
    "Incident_Category_Vision",
]

# ── Blue Night Network ─────────────────────────────────────────────────────────
# TTC overnight service runs ~1:30am–5:30am on 300-series routes only.
# All other routes are suspended during this window.
BLUE_NIGHT_HOURS  = {1, 2, 3, 4, 5}
BLUE_NIGHT_ROUTES = {
    300, 301, 302, 303, 304, 306, 307, 308, 309, 310,
    312, 313, 314, 320, 322, 324, 325, 329, 330, 332,
    334, 335, 336, 337, 340, 341, 342, 343, 344, 345,
    352, 353, 354, 385,
}


def validate_route_time(route: int, hour: int) -> tuple[bool, str]:
    """Returns (is_valid, error_message). Empty message means valid."""
    if hour in BLUE_NIGHT_HOURS:
        if route not in BLUE_NIGHT_ROUTES:
            overnight_list = ", ".join(str(r) for r in sorted(BLUE_NIGHT_ROUTES))
            return False, (
                f"\n  ✗  Route {route} does not operate between 1am–5am.\n"
                f"     Only TTC Blue Night Network routes run overnight:\n"
                f"     {overnight_list}\n"
                f"     Try a different route or a time outside 01:00–05:59."
            )
    return True, ""


def get_season(month: int) -> str:
    if month in [12, 1, 2]:  return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    else:                     return "Fall"


def is_rush_hour(hour: int) -> int:
    return 1 if hour in [7, 8, 9, 16, 17, 18] else 0


def build_feature_row(route, dt, temp, visibility, wind, humidity) -> np.ndarray:
    hour     = dt.hour
    month    = dt.month
    day_name = dt.strftime("%A")
    season   = get_season(month)

    row = {col: 0 for col in FEATURE_COLUMNS}

    row["Route_Number"]     = route
    row["Hour"]             = hour
    row["Month"]            = month
    row["Is_Rush_Hour"]     = is_rush_hour(hour)
    row["Is_Weekend"]       = 1 if day_name in ["Saturday", "Sunday"] else 0
    row["Temp_C"]           = temp
    row["Visibility_km"]    = visibility
    row["Wind_Spd_kmh"]     = wind
    row["Rel_Humidity_pct"] = humidity

    day_col = f"Day_{day_name}"
    if day_col in row:
        row[day_col] = 1

    season_col = f"Season_{season}"
    if season_col in row:
        row[season_col] = 1

    # All Incident_Category columns left as 0 — unknown before the delay occurs

    return np.array([row[col] for col in FEATURE_COLUMNS]).reshape(1, -1)


def prompt_float(label, default):
    raw = input(f"  {label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  ✗  Invalid — using default ({default})")
        return default


def main():
    print("\n" + "="*60)
    print("  TTC Bus Delay — Severity Predictor")
    print("  Model: Random Forest  |  Trained on 2015–2023 data")
    print("  Predicts whether a delay will be SEVERE (≥ 15 min)")
    print("="*60)

    if not os.path.exists(MODEL_PATH):
        print(f"\n  ERROR: Model not found at:\n  {MODEL_PATH}")
        print("  Run train_model.py first.")
        return

    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    print(f"\n  Model loaded.\n")

    while True:
        print("-"*60)
        print("  Enter scenario details\n")

        # Route
        while True:
            raw = input("  Route Number (e.g. 29, 504, 192): ").strip()
            try:
                route = int(raw)
                break
            except ValueError:
                print("  ✗  Enter a numeric route number.")

        # Date & time
        while True:
            raw_dt = input("  Date & Time (YYYY-MM-DD HH:MM): ").strip()
            try:
                dt = datetime.strptime(raw_dt, "%Y-%m-%d %H:%M")
                break
            except ValueError:
                print("  ✗  Use format YYYY-MM-DD HH:MM")

        # ── Blue Night validation ─────────────────────────────────────────────
        valid, err_msg = validate_route_time(route, dt.hour)
        if not valid:
            print(err_msg)
            again = input("  Try a different scenario? (y/n) [y]: ").strip().lower()
            if again == "n":
                break
            continue

        # Weather
        print()
        temp       = prompt_float("Temperature (°C)",      5.0)
        visibility = prompt_float("Visibility (km)",       15.0)
        wind       = prompt_float("Wind Speed (km/h)",     20.0)
        humidity   = prompt_float("Relative Humidity (%)", 70.0)

        # Derived values
        season  = get_season(dt.month)
        rush    = is_rush_hour(dt.hour)
        weekend = dt.strftime("%A") in ["Saturday", "Sunday"]

        print(f"\n  ── Scenario Summary ────────────────────────────")
        print(f"     Route:      {route}")
        print(f"     Date/Time:  {dt.strftime('%A %b %d %Y, %H:%M')}")
        print(f"     Season:     {season}  |  Rush Hour: {'Yes' if rush else 'No'}  |  Weekend: {'Yes' if weekend else 'No'}")
        print(f"     Weather:    {temp}°C, {visibility} km visibility, {wind} km/h wind, {humidity}% humidity")

        X    = build_feature_row(route, dt, temp, visibility, wind, humidity)
        pred = clf.predict(X)[0]
        prob = clf.predict_proba(X)[0]

        print(f"\n  ── Prediction ──────────────────────────────────")
        if pred == 1:
            print(f"     Result:  ⚠  SEVERE DELAY LIKELY  (≥ 15 min)")
        else:
            print(f"     Result:  ✓  Delay unlikely to be severe  (< 15 min)")
        print(f"     Probability — Not Severe: {prob[0]*100:.1f}%  |  Severe: {prob[1]*100:.1f}%")
        print()

        again = input("  Run another scenario? (y/n) [y]: ").strip().lower()
        if again == "n":
            break

    print("\n  Done.\n")


if __name__ == "__main__":
    main()