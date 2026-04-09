"""
build_deploy_data.py — Pre-aggregation script for deployment
Reads master_ttc_eda_master.csv and produces small summary CSVs for deploy/data/
Run with: python build_deploy_data.py  (from analysis/ folder)
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(SCRIPT_DIR)
INPUT_FILE  = os.path.join(ROOT, "data", "processed", "master_ttc_eda_master.csv")
OUTPUT_DIR  = os.path.join(ROOT, "deploy", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"build_deploy_data started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Loading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["Service_Date"] = pd.to_datetime(df["Service_Date"], errors="coerce")
df_del = df[df["Min_Delay"] > 0].copy()
print(f"Loaded: {len(df):,} rows  |  Delay rows: {len(df_del):,}")

# ── 1. KPI summary ────────────────────────────────────────────────────────────
ly = df_del[df_del["Year"] == df_del["Year"].max()]
py = df_del[df_del["Year"] == df_del["Year"].max() - 1]

kpi = pd.DataFrame([{
    "total_incidents":  len(df_del),
    "avg_delay":        round(df_del["Min_Delay"].mean(), 2),
    "severe_rate":      round(df_del["Is_Severe"].mean() * 100, 1),
    "severe_30_rate":   round(df_del["Is_Severe_30"].mean() * 100, 1),
    "yoy_incident_pct": round((len(ly) - len(py)) / len(py) * 100, 1) if len(py) else 0,
    "yoy_delay_min":    round(ly["Min_Delay"].mean() - py["Min_Delay"].mean(), 2),
    "max_year":         int(df_del["Year"].max()),
}])
kpi.to_csv(os.path.join(OUTPUT_DIR, "kpi.csv"), index=False)
print(f"  kpi.csv — {len(kpi)} rows")

# ── 2. Year-over-year trend ───────────────────────────────────────────────────
yoy = df_del.groupby("Year").agg(
    Incidents=("Min_Delay","count"),
    Avg_Delay=("Min_Delay","mean"),
    Severe_Rate=("Is_Severe","mean"),
    Severe_30_Rate=("Is_Severe_30","mean"),
).round(3).reset_index()
yoy.to_csv(os.path.join(OUTPUT_DIR, "yoy.csv"), index=False)
print(f"  yoy.csv — {len(yoy)} rows")

# ── 3. Monthly patterns ───────────────────────────────────────────────────────
monthly = df_del.groupby("Month").agg(
    Avg_Delay=("Min_Delay","mean"),
    Severe_Rate=("Is_Severe","mean"),
    Incidents=("Min_Delay","count"),
).round(3).reset_index()
monthly["Month_Name"] = monthly["Month"].map({
    1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
monthly.to_csv(os.path.join(OUTPUT_DIR, "monthly.csv"), index=False)
print(f"  monthly.csv — {len(monthly)} rows")

# ── 4. Route summary ──────────────────────────────────────────────────────────
route = (
    df_del.groupby("Route_Number")
    .agg(
        Incidents=("Min_Delay","count"),
        Avg_Delay=("Min_Delay","mean"),
        Severe_Rate=("Is_Severe","mean"),
        Avg_Headway=("Headway_min","mean"),
    )
    .round(3).query("Incidents >= 100").reset_index()
)
route["Risk_Score"] = (
    route["Avg_Delay"] * route["Avg_Headway"] * route["Severe_Rate"]
).round(2)
route.to_csv(os.path.join(OUTPUT_DIR, "route_summary.csv"), index=False)
print(f"  route_summary.csv — {len(route)} rows")

# ── 5. Peak period route risk ─────────────────────────────────────────────────
peak = df_del[df_del["Is_Rush_Hour"] == 1]
peak_route = (
    peak.groupby("Route_Number")
    .agg(
        Peak_Incidents=("Min_Delay","count"),
        Avg_Delay_Peak=("Min_Delay","mean"),
        Severe_Rate_Peak=("Is_Severe","mean"),
        Severe_30_Rate_Peak=("Is_Severe_30","mean"),
    )
    .round(3).query("Peak_Incidents >= 50").reset_index()
)
peak_route.to_csv(os.path.join(OUTPUT_DIR, "peak_route.csv"), index=False)
print(f"  peak_route.csv — {len(peak_route)} rows")

# ── 6. Incident summary ───────────────────────────────────────────────────────
inc = (
    df_del.dropna(subset=["Incident_Category"])
    .groupby("Incident_Category")
    .agg(
        Total=("Min_Delay","count"),
        Avg_Delay=("Min_Delay","mean"),
        Severe_Rate=("Is_Severe","mean"),
        Total_Hrs=("Min_Delay", lambda x: x.sum()/60),
    )
    .round(2).sort_values("Total", ascending=False).reset_index()
)
inc["Share"] = (inc["Total"] / inc["Total"].sum() * 100).round(1)
inc.to_csv(os.path.join(OUTPUT_DIR, "incident_summary.csv"), index=False)
print(f"  incident_summary.csv — {len(inc)} rows")

# ── 7. Vehicle intel ──────────────────────────────────────────────────────────
mech = df_del[
    (df_del["Incident_Category"] == "Mechanical") &
    (df_del["Vehicle"].notna()) &
    (df_del["Vehicle"].astype(str) != "0.0")
].copy()
vi = (
    mech.groupby("Vehicle")
    .agg(
        Incidents=("Min_Delay","count"),
        Avg_Delay=("Min_Delay","mean"),
        Total_Hrs=("Min_Delay", lambda x: x.sum()/60),
        Severe_Rate=("Is_Severe","mean"),
    )
    .round(2).sort_values("Incidents", ascending=False).reset_index().head(30)
)
vi["Total_Vehicles"] = mech["Vehicle"].nunique()
vi.to_csv(os.path.join(OUTPUT_DIR, "vehicle_intel.csv"), index=False)
print(f"  vehicle_intel.csv — {len(vi)} rows")

# ── 8. Weather — temperature ──────────────────────────────────────────────────
dw = df_del.dropna(subset=["Temp_C"]).copy()
dw["Temp_Band"] = pd.cut(dw["Temp_C"],
    bins=[-35,-15,0,5,15,25,45],
    labels=["Extreme Cold (<-15°C)","Cold (-15–0°C)","Cool (0–5°C)",
            "Mild (5–15°C)","Warm (15–25°C)","Hot (25°C+)"])
temp_agg = dw.groupby("Temp_Band", observed=True).agg(
    Avg_Delay=("Min_Delay","mean"),
    Severe_Rate=("Is_Severe","mean"),
    Count=("Min_Delay","count"),
).round(3).reset_index()
temp_agg.to_csv(os.path.join(OUTPUT_DIR, "weather_temp.csv"), index=False)
print(f"  weather_temp.csv — {len(temp_agg)} rows")

# ── 9. Weather — visibility ───────────────────────────────────────────────────
dv = df_del.dropna(subset=["Visibility_km"]).copy()
dv["Vis_Band"] = pd.cut(dv["Visibility_km"],
    bins=[0,1,5,15,100], labels=["<1km","1–5km","5–15km","15km+"])
vis_agg = dv.groupby("Vis_Band", observed=True).agg(
    Avg_Delay=("Min_Delay","mean"),
    Severe_Rate=("Is_Severe","mean"),
    Count=("Min_Delay","count"),
).round(3).reset_index()
vis_agg.to_csv(os.path.join(OUTPUT_DIR, "weather_vis.csv"), index=False)
print(f"  weather_vis.csv — {len(vis_agg)} rows")

# ── 10. Weather — wind ────────────────────────────────────────────────────────
dw2 = df_del.dropna(subset=["Wind_Spd_kmh"]).copy()
dw2["Wind_Band"] = pd.cut(dw2["Wind_Spd_kmh"],
    bins=[0,10,20,30,50,150], labels=["0–10","10–20","20–30","30–50","50+"])
wind_agg = dw2.groupby("Wind_Band", observed=True).agg(
    Severe_Rate=("Is_Severe","mean"),
    Count=("Min_Delay","count"),
).round(3).reset_index()
wind_agg.to_csv(os.path.join(OUTPUT_DIR, "weather_wind.csv"), index=False)
print(f"  weather_wind.csv — {len(wind_agg)} rows")

# ── 11. Seasonal ──────────────────────────────────────────────────────────────
seas = df_del.groupby("Season").agg(
    Avg_Delay=("Min_Delay","mean"),
    Severe_Rate=("Is_Severe","mean"),
    Incidents=("Min_Delay","count"),
).round(3).reset_index()
seas.to_csv(os.path.join(OUTPUT_DIR, "seasonal.csv"), index=False)
print(f"  seasonal.csv — {len(seas)} rows")

# ── 12. Neighbourhood density (if available) ──────────────────────────────────
if "Neighbourhood" in df_del.columns and df_del["Neighbourhood"].notna().sum() > 0:
    nbhd = (
        df_del[df_del["Neighbourhood"].notna()]
        .groupby("Neighbourhood")
        .agg(
            Incidents=("Min_Delay","count"),
            Avg_Delay=("Min_Delay","mean"),
            Severe_Rate=("Is_Severe","mean"),
            Population_Density=("Population_Density","first"),
            Avg_Headway=("Headway_min","mean"),
        )
        .round(2).dropna(subset=["Population_Density"]).reset_index()
    )
    nbhd["Incidents_Per_Density"] = (nbhd["Incidents"] / nbhd["Population_Density"]).round(4)
    nbhd.to_csv(os.path.join(OUTPUT_DIR, "neighbourhood.csv"), index=False)
    print(f"  neighbourhood.csv — {len(nbhd)} rows")
else:
    print(f"  neighbourhood.csv — skipped (no geocoded data)")

# ── Size report ───────────────────────────────────────────────────────────────
print(f"\n── Output files ──")
total_kb = 0
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".csv"):
        kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        total_kb += kb
        print(f"  {f:<35} {kb:>8.1f} KB")
print(f"  {'TOTAL':<35} {total_kb/1024:>7.2f} MB")
print(f"\nbuild_deploy_data complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Next: copy deploy/ folder to GitHub repo and deploy to Streamlit Cloud")