"""
app.py — TTC Bus Delay Severity Predictor (Streamlit)
Run with: streamlit run app.py
"""
import os
import pickle
import numpy as np
import streamlit as st
from datetime import datetime, date, time

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# Try to find the model in the current directory or one level up
BUNDLE_PATH = os.path.join(SCRIPT_DIR, "models", "lgbm_ttc_bundle.pkl")
if not os.path.exists(BUNDLE_PATH):
    BUNDLE_PATH = os.path.join(SCRIPT_DIR, "..", "models", "lgbm_ttc_bundle.pkl")


# ── Blue Night validation ──────────────────────────────────────────────────────
BLUE_NIGHT_HOURS  = {1, 2, 3, 4, 5}
BLUE_NIGHT_ROUTES = {
    300, 301, 302, 303, 304, 306, 307, 308, 309, 310,
    312, 313, 314, 320, 322, 324, 325, 329, 330, 332,
    334, 335, 336, 337, 340, 341, 342, 343, 344, 345,
    352, 353, 354, 385,
}


def get_season(month: int) -> str:
    if month in [12, 1, 2]:  return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    else:                     return "Fall"


def is_rush_hour(hour: int) -> int:
    return 1 if hour in [7, 8, 9, 16, 17, 18] else 0


def build_feature_row(route, dt, temp, visibility, wind, humidity,
                       route_encoding, global_rate, feature_columns) -> np.ndarray:
    hour     = dt.hour
    month    = dt.month
    day_name = dt.strftime("%A")
    season   = get_season(month)
    rush     = is_rush_hour(hour)
    weekend  = 1 if day_name in ["Saturday", "Sunday"] else 0

    route_rate = route_encoding.get(route, global_rate)

    row = {col: 0 for col in feature_columns}
    row["Hour"]              = hour
    row["Month"]             = month
    row["Is_Rush_Hour"]      = rush
    row["Is_Weekend"]        = weekend
    row["Temp_C"]            = temp
    row["Visibility_km"]     = visibility
    row["Wind_Spd_kmh"]      = wind
    row["Rel_Humidity_pct"]  = humidity
    row["Route_Severe_Rate"] = route_rate
    row["Rush_x_Route_Rate"] = rush * route_rate
    row["Hour_x_Month"]      = hour * month

    day_col = f"Day_{day_name}"
    if day_col in row:
        row[day_col] = 1

    season_col = f"Season_{season}"
    if season_col in row:
        row[season_col] = 1

    return np.array([row[col] for col in feature_columns]).reshape(1, -1)


@st.cache_resource
def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTC Delay Predictor",
    page_icon="🚌",
    layout="centered",
)

st.title("🚌 TTC Bus Delay — Severity Predictor")
st.caption("Predicts whether a bus delay will be **severe (≥ 15 min)** based on route, time, and weather conditions.")
st.divider()

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(BUNDLE_PATH):
    st.error(f"Model bundle not found at:\n`{BUNDLE_PATH}`\n\nRun `train_model_v2.py` first.")
    st.stop()

bundle          = load_bundle(BUNDLE_PATH)
clf             = bundle["model"]
route_encoding  = bundle["route_encoding"]
global_rate     = bundle["global_rate"]
feature_columns = bundle["feature_columns"]
known_routes    = sorted(route_encoding.keys())

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    route = st.number_input("Route Number", min_value=1, max_value=999, value=29, step=1)
    if route not in route_encoding:
        st.caption("⚠️ Route not seen in training data — using global average rate.")

with col2:
    selected_date = st.date_input("Date", value=date.today())
    selected_time = st.time_input("Time", value=time(8, 0), step=1800)

st.subheader("Weather Conditions")
w1, w2 = st.columns(2)
with w1:
    temp       = st.slider("Temperature (°C)",      min_value=-30, max_value=40,  value=5)
    visibility = st.slider("Visibility (km)",        min_value=0,   max_value=50,  value=15)
with w2:
    wind       = st.slider("Wind Speed (km/h)",      min_value=0,   max_value=100, value=20)
    humidity   = st.slider("Relative Humidity (%)",  min_value=0,   max_value=100, value=70)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Predict Severity", type="primary", use_container_width=True):
    dt   = datetime.combine(selected_date, selected_time)
    hour = dt.hour

    # Blue Night validation
    if hour in BLUE_NIGHT_HOURS and route not in BLUE_NIGHT_ROUTES:
        st.error(
            f"**Route {route} does not operate between 1am–5am.**\n\n"
            f"Only TTC Blue Night Network routes (300-series) run overnight. "
            f"Try a time outside 01:00–05:59, or use a Blue Night route."
        )
    else:
        season  = get_season(dt.month)
        rush    = is_rush_hour(hour)
        weekend = dt.strftime("%A") in ["Saturday", "Sunday"]

        X    = build_feature_row(route, dt, temp, visibility, wind, humidity,
                                  route_encoding, global_rate, feature_columns)
        prob = clf.predict_proba(X)[0]
        pred = clf.predict(X)[0]

        prob_not = prob[0] * 100
        prob_sev = prob[1] * 100

        # Result banner
        if pred == 1:
            st.error(f"### ⚠️ Severe Delay Likely (≥ 15 min)")
        else:
            st.success(f"### ✅ Delay Unlikely to be Severe (< 15 min)")

        # Probability bar
        st.markdown("**Prediction Confidence**")
        st.progress(int(prob_sev), text=f"Severe: {prob_sev:.1f}%  |  Not Severe: {prob_not:.1f}%")

        # Scenario summary
        with st.expander("Scenario details"):
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.metric("Route", route)
                st.metric("Date / Time", dt.strftime("%a %b %d %Y, %H:%M"))
                st.metric("Season", season)
            with detail_col2:
                st.metric("Rush Hour", "Yes" if rush else "No")
                st.metric("Weekend", "Yes" if weekend else "No")
                st.metric("Temperature", f"{temp}°C")