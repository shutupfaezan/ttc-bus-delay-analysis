"""
eda_analysis.py — TTC Bus Delay EDA Validation Tables
Produces ground-truth summary tables for all PBIs.
Use to validate Power BI / Tableau visualisations.
Run with: python eda_analysis.py  (from analysis/ folder)
Requires: master_ttc_eda_master.csv to exist (run prepare_eda_master.py first)
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE = "../data/processed/master_ttc_eda_master.csv"
LOG_FILE   = "../logs/eda_analysis_log.txt"
SEP        = "=" * 70
# ─────────────────────────────────────────────────────────────────────────────

log_lines = []
def log(text=""):
    print(text)
    log_lines.append(str(text))

def section(title):
    log(f"\n{SEP}")
    log(f"  {title}")
    log(SEP)

log(f"EDA Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── LOAD ──────────────────────────────────────────────────────────────────────
log(f"\nLoading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, low_memory=False)
log(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
log(f"Years covered: {df['Year'].min():.0f} – {df['Year'].max():.0f}")

# Work only with rows that have a delay
df_delay = df[df['Min_Delay'] > 0].copy()
log(f"Rows with delay > 0: {len(df_delay):,}")

# ── PBI 1 — TOP ROUTES BY FREQUENCY AND AVG DELAY OVER 10 YEARS ──────────────
section("PBI 1 — Top 30 routes: frequency and avg delay (2015–2025)")
route_summary = (
    df_delay.groupby('Route_Number')
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Min    = ('Min_Delay', 'mean'),
        Median_Delay_Min = ('Min_Delay', 'median'),
        Total_Delay_Hrs  = ('Min_Delay', lambda x: x.sum() / 60),
        Severe_Rate_Pct  = ('Is_Severe', 'mean'),
        Severe_30_Rate   = ('Is_Severe_30', 'mean'),
    )
    .round(2)
    .sort_values('Total_Incidents', ascending=False)
    .reset_index()
    .head(30)
)
route_summary['Severe_Rate_Pct'] = (route_summary['Severe_Rate_Pct'] * 100).round(1)
route_summary['Severe_30_Rate']  = (route_summary['Severe_30_Rate']  * 100).round(1)
route_summary['Total_Delay_Hrs'] = route_summary['Total_Delay_Hrs'].round(0).astype(int)
log(route_summary.to_string(index=False))

# ── PBI 1b — TOP ROUTES BY AVERAGE DELAY DURATION ────────────────────────────
section("PBI 1b — Top 20 routes by average delay duration")
route_avg = (
    df_delay[df_delay['Total_Incidents'] if 'Total_Incidents' in df_delay.columns
             else df_delay['Route_Number'].notna()]
    .groupby('Route_Number')
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Min    = ('Min_Delay', 'mean'),
        Median_Delay_Min = ('Min_Delay', 'median'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
    )
    .round(2)
    .query('Total_Incidents >= 100')   # minimum 100 incidents for statistical validity
    .sort_values('Avg_Delay_Min', ascending=False)
    .reset_index()
    .head(20)
)
log(route_avg.to_string(index=False))

# ── SEVERE RATE — % OF DELAYS THAT ARE SEVERE PER ROUTE ──────────────────────
section("Severe Rate — Top 20 routes by % of severe delays (>=15 min, min 100 incidents)")
severe_rate = (
    df_delay.groupby('Route_Number')
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
        Severe_30_Pct    = ('Is_Severe_30', lambda x: x.mean() * 100),
        Avg_Delay_Min    = ('Min_Delay', 'mean'),
    )
    .round(2)
    .query('Total_Incidents >= 100')
    .sort_values('Severe_Rate_Pct', ascending=False)
    .reset_index()
    .head(20)
)
log(severe_rate.to_string(index=False))

# ── YoY DELAY TREND ───────────────────────────────────────────────────────────
section("Year-over-Year Delay Trend (2015–2025)")
yoy = (
    df_delay.groupby('Year')
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Min    = ('Min_Delay', 'mean'),
        Median_Delay_Min = ('Min_Delay', 'median'),
        Total_Delay_Hrs  = ('Min_Delay', lambda x: x.sum() / 60),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
        Severe_30_Rate   = ('Is_Severe_30', lambda x: x.mean() * 100),
        Unique_Routes    = ('Route_Number', 'nunique'),
    )
    .round(2)
    .reset_index()
)
yoy['Total_Delay_Hrs'] = yoy['Total_Delay_Hrs'].round(0).astype(int)
log(yoy.to_string(index=False))

# ── PBI 3 — MONTHLY DELAY PATTERNS ───────────────────────────────────────────
section("PBI 3 — Monthly delay patterns (avg across all years)")
monthly = (
    df_delay.groupby('Month')
    .agg(
        Avg_Incidents_Per_Year = ('Min_Delay', lambda x: len(x) / df_delay['Year'].nunique()),
        Avg_Delay_Min          = ('Min_Delay', 'mean'),
        Severe_Rate_Pct        = ('Is_Severe', lambda x: x.mean() * 100),
        Severe_30_Rate         = ('Is_Severe_30', lambda x: x.mean() * 100),
    )
    .round(2)
    .reset_index()
)
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
monthly['Month_Name'] = monthly['Month'].map(month_names)
log(monthly.to_string(index=False))

section("PBI 3b — Monthly totals by year (for time series projection)")
monthly_yoy = (
    df_delay.groupby(['Year', 'Month'])
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Min    = ('Min_Delay', 'mean'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
    )
    .round(2)
    .reset_index()
)
log(monthly_yoy.to_string(index=False))

# ── INCIDENT TYPE WEIGHT (TREEMAP DATA) ───────────────────────────────────────
section("Incident Type Weight — for treemap")
incident = (
    df_delay.groupby('Incident_Category')
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Min    = ('Min_Delay', 'mean'),
        Median_Delay_Min = ('Min_Delay', 'median'),
        Total_Delay_Hrs  = ('Min_Delay', lambda x: x.sum() / 60),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
        Severe_30_Rate   = ('Is_Severe_30', lambda x: x.mean() * 100),
    )
    .round(2)
    .sort_values('Total_Incidents', ascending=False)
    .reset_index()
)
incident['Total_Delay_Hrs']  = incident['Total_Delay_Hrs'].round(0).astype(int)
incident['Share_of_Total_Pct'] = (
    incident['Total_Incidents'] / incident['Total_Incidents'].sum() * 100
).round(1)
log(incident.to_string(index=False))

# ── VEHICLE INTEL ─────────────────────────────────────────────────────────────
section("Vehicle Intel — worst buses by mechanical incidents")
mech_df = df_delay[
    (df_delay['Incident_Category'] == 'Mechanical') &
    (df_delay['Vehicle'].notna())
].copy()

# Filter out placeholder 0.0 vehicle IDs
mech_df = mech_df[mech_df['Vehicle'].astype(str) != '0.0']

vehicle_intel = (
    mech_df.groupby('Vehicle')
    .agg(
        Mechanical_Incidents = ('Min_Delay', 'count'),
        Avg_Delay_Min        = ('Min_Delay', 'mean'),
        Total_Delay_Hrs      = ('Min_Delay', lambda x: x.sum() / 60),
        Severe_Rate_Pct      = ('Is_Severe', lambda x: x.mean() * 100),
        Years_Active         = ('Year', lambda x: f"{int(x.min())}–{int(x.max())}"),
    )
    .round(2)
    .sort_values('Mechanical_Incidents', ascending=False)
    .reset_index()
    .head(30)
)
vehicle_intel['Total_Delay_Hrs'] = vehicle_intel['Total_Delay_Hrs'].round(1)
log(f"Total vehicles with mechanical incidents: {mech_df['Vehicle'].nunique():,}")
log(vehicle_intel.to_string(index=False))

# ── SEASONAL PATTERNS ─────────────────────────────────────────────────────────
section("Seasonal delay patterns")
seasonal = (
    df_delay.groupby(['Season', 'Year'])
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Min    = ('Min_Delay', 'mean'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
    )
    .round(2)
    .reset_index()
    .sort_values(['Year', 'Season'])
)
log(seasonal.to_string(index=False))

section("Seasonal avg across all years")
seasonal_avg = (
    df_delay.groupby('Season')
    .agg(
        Avg_Incidents_Per_Year = ('Min_Delay', lambda x: len(x) / df_delay['Year'].nunique()),
        Avg_Delay_Min          = ('Min_Delay', 'mean'),
        Severe_Rate_Pct        = ('Is_Severe', lambda x: x.mean() * 100),
        Severe_30_Rate         = ('Is_Severe_30', lambda x: x.mean() * 100),
    )
    .round(2)
    .reset_index()
)
log(seasonal_avg.to_string(index=False))

# ── WEATHER IMPACT ────────────────────────────────────────────────────────────
section("Weather impact — temperature bands")
df_delay['Temp_Band'] = pd.cut(
    df_delay['Temp_C'],
    bins=[-30, -15, 0, 5, 15, 25, 40],
    labels=['Extreme Cold (<-15)', 'Cold (-15 to 0)', 'Cool (0–5)',
            'Mild (5–15)', 'Warm (15–25)', 'Hot (25+)']
)
temp_impact = (
    df_delay.groupby('Temp_Band', observed=True)
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Min    = ('Min_Delay', 'mean'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
        Severe_30_Rate   = ('Is_Severe_30', lambda x: x.mean() * 100),
    )
    .round(2)
    .reset_index()
)
log(temp_impact.to_string(index=False))

section("Weather impact — visibility bands")
df_delay['Vis_Band'] = pd.cut(
    df_delay['Visibility_km'],
    bins=[0, 1, 5, 15, 100],
    labels=['Near Zero (<1km)', 'Poor (1–5km)', 'Moderate (5–15km)', 'Clear (15km+)']
)
vis_impact = (
    df_delay.groupby('Vis_Band', observed=True)
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Min    = ('Min_Delay', 'mean'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
    )
    .round(2)
    .reset_index()
)
log(vis_impact.to_string(index=False))

# ── ROUTE RISK PROFILE — HIGH DELAY + LOW FREQUENCY ──────────────────────────
section("Route risk profile — high delay density + low service frequency")
if 'Headway_min' in df_delay.columns:
    route_risk = (
        df_delay.groupby('Route_Number')
        .agg(
            Total_Incidents  = ('Min_Delay', 'count'),
            Avg_Delay_Min    = ('Min_Delay', 'mean'),
            Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
            Avg_Headway_Min  = ('Headway_min', 'mean'),
        )
        .round(2)
        .query('Total_Incidents >= 100')
        .reset_index()
    )
    # Risk score: higher delay avg × higher headway × higher severe rate
    route_risk['Risk_Score'] = (
        route_risk['Avg_Delay_Min'] *
        route_risk['Avg_Headway_Min'] *
        route_risk['Severe_Rate_Pct'] / 100
    ).round(2)
    route_risk = route_risk.sort_values('Risk_Score', ascending=False).head(25)
    log(f"Routes scored on: Avg_Delay × Headway × Severe_Rate")
    log(f"Higher score = more operationally risky for passengers")
    log(route_risk.to_string(index=False))
else:
    log("  Headway_min column not found — skipping route risk profile")

# ── NEIGHBOURHOOD ANALYSIS (if geocoding complete) ───────────────────────────
section("Neighbourhood delay analysis (requires geocoding)")
if 'Neighbourhood' in df_delay.columns and df_delay['Neighbourhood'].notna().sum() > 100:
    nbhd = (
        df_delay[df_delay['Neighbourhood'].notna()]
        .groupby('Neighbourhood')
        .agg(
            Total_Incidents      = ('Min_Delay', 'count'),
            Avg_Delay_Min        = ('Min_Delay', 'mean'),
            Severe_Rate_Pct      = ('Is_Severe', lambda x: x.mean() * 100),
            Population_Density   = ('Population_Density', 'first'),
            Avg_Headway_Min      = ('Headway_min', 'mean'),
        )
        .round(2)
        .dropna(subset=['Population_Density'])
        .reset_index()
    )
    # Frequency-to-density ratio: incidents per 1000 people per km²
    nbhd['Incidents_Per_Density'] = (
        nbhd['Total_Incidents'] / nbhd['Population_Density']
    ).round(4)
    nbhd = nbhd.sort_values('Total_Incidents', ascending=False).head(30)
    log(nbhd.to_string(index=False))
else:
    log("  Neighbourhood data not available yet.")
    log("  Run geocode_nominatim_filtered.py then prepare_eda_master.py first.")

# ── PEAK PERIOD RISK — PBI 2 ──────────────────────────────────────────────────
section("PBI 2 — Routes most likely to experience delays in peak periods")
peak_df = df_delay[df_delay['Is_Rush_Hour'] == 1].copy()
peak_risk = (
    peak_df.groupby('Route_Number')
    .agg(
        Peak_Incidents       = ('Min_Delay', 'count'),
        Avg_Delay_Peak       = ('Min_Delay', 'mean'),
        Severe_Rate_Peak_Pct = ('Is_Severe', lambda x: x.mean() * 100),
        Severe_30_Peak_Pct   = ('Is_Severe_30', lambda x: x.mean() * 100),
    )
    .round(2)
    .query('Peak_Incidents >= 50')
    .sort_values('Peak_Incidents', ascending=False)
    .reset_index()
    .head(25)
)
log(f"Peak period = rush hours (7–9am, 4–6pm)")
log(peak_risk.to_string(index=False))

section("PBI 2b — Winter peak period (highest risk combination)")
winter_peak = df_delay[
    (df_delay['Is_Rush_Hour'] == 1) &
    (df_delay['Season'] == 'Winter')
].copy()
winter_risk = (
    winter_peak.groupby('Route_Number')
    .agg(
        Winter_Peak_Incidents = ('Min_Delay', 'count'),
        Avg_Delay_Min         = ('Min_Delay', 'mean'),
        Severe_Rate_Pct       = ('Is_Severe', lambda x: x.mean() * 100),
        Severe_30_Rate        = ('Is_Severe_30', lambda x: x.mean() * 100),
    )
    .round(2)
    .query('Winter_Peak_Incidents >= 30')
    .sort_values('Severe_30_Rate', ascending=False)
    .reset_index()
    .head(20)
)
log(f"Winter rush hour routes — ranked by 30+ min severe rate")
log(winter_risk.to_string(index=False))

# ── SAVE LOG ──────────────────────────────────────────────────────────────────
log(f"\n{SEP}")
log(f"EDA Analysis complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"{SEP}")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Full output saved to: {LOG_FILE}")