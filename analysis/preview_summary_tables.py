import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE = "../data/processed/master_ttc_eda_ready.csv"
# ─────────────────────────────────────────────────────────────────────────────

print("Loading master_ttc_eda_ready.csv...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
df['Service_DateTime'] = pd.to_datetime(df['Service_DateTime'], errors='coerce')
df_nonzero = df[df['Min_Delay'] > 0]   # exclude zero-delay rows for averages
print(f"Loaded: {len(df):,} rows\n")

SEP = "=" * 65

# ── 1. ROUTE PERFORMANCE ──────────────────────────────────────────────────────
print(SEP)
print("1. ROUTE PERFORMANCE (top 15 routes)")
print(SEP)
route = (df_nonzero.groupby('Route_Number')
    .agg(
        Total_Incidents   = ('Min_Delay', 'count'),
        Total_Delay_Mins  = ('Min_Delay', 'sum'),
        Avg_Delay_Mins    = ('Min_Delay', 'mean'),
        Severe_Incidents  = ('Is_Severe', 'sum'),
        Severe_Rate_Pct   = ('Is_Severe', lambda x: x.mean() * 100),
    )
    .round(1)
    .sort_values('Total_Delay_Mins', ascending=False)
    .reset_index()
    .head(15))
print(route.to_string(index=False))

# ── 2. YEAR OVER YEAR TRENDS ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("2. YEAR-OVER-YEAR TRENDS")
print(SEP)
yearly = (df_nonzero.groupby('Year')
    .agg(
        Total_Incidents   = ('Min_Delay', 'count'),
        Total_Delay_Mins  = ('Min_Delay', 'sum'),
        Avg_Delay_Mins    = ('Min_Delay', 'mean'),
        Severe_Incidents  = ('Is_Severe', 'sum'),
        Severe_Rate_Pct   = ('Is_Severe', lambda x: x.mean() * 100),
        Unique_Routes     = ('Route_Number', 'nunique'),
    )
    .round(1)
    .reset_index())
print(yearly.to_string(index=False))

# ── 3. INCIDENT TYPE BREAKDOWN ────────────────────────────────────────────────
print(f"\n{SEP}")
print("3. INCIDENT TYPE BREAKDOWN (top 20)")
print(SEP)
incidents = (df_nonzero.groupby('Incident_Code')
    .agg(
        Total_Incidents   = ('Min_Delay', 'count'),
        Total_Delay_Mins  = ('Min_Delay', 'sum'),
        Avg_Delay_Mins    = ('Min_Delay', 'mean'),
        Severe_Rate_Pct   = ('Is_Severe', lambda x: x.mean() * 100),
    )
    .round(1)
    .sort_values('Total_Delay_Mins', ascending=False)
    .reset_index()
    .head(20))
print(incidents.to_string(index=False))

# ── 4. WEATHER IMPACT ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("4. WEATHER IMPACT — TEMPERATURE BINS")
print(SEP)
df_nonzero = df_nonzero.copy()
df_nonzero['Temp_Band'] = pd.cut(
    df_nonzero['Temp_C'],
    bins=[-30, -15, 0, 5, 15, 25, 40],
    labels=['Extreme Cold (<-15)', 'Cold (-15 to 0)', 'Cool (0-5)',
            'Mild (5-15)', 'Warm (15-25)', 'Hot (25+)']
)
weather_temp = (df_nonzero.groupby('Temp_Band', observed=True)
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Mins   = ('Min_Delay', 'mean'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
    )
    .round(1)
    .reset_index())
print(weather_temp.to_string(index=False))

print(f"\n── Visibility Bands ──")
df_nonzero['Vis_Band'] = pd.cut(
    df_nonzero['Visibility_km'],
    bins=[0, 1, 5, 15, 100],
    labels=['Near Zero (<1km)', 'Poor (1-5km)', 'Moderate (5-15km)', 'Clear (15km+)']
)
weather_vis = (df_nonzero.groupby('Vis_Band', observed=True)
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Mins   = ('Min_Delay', 'mean'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
    )
    .round(1)
    .reset_index())
print(weather_vis.to_string(index=False))

# ── 5. TIME OF DAY PATTERNS ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("5. TIME OF DAY PATTERNS")
print(SEP)
time_day = (df_nonzero.groupby(['Hour', 'Day'])
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Mins   = ('Min_Delay', 'mean'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
    )
    .round(1)
    .reset_index())
print("Sample (first 14 rows — all 7 days for hours 7 and 8):")
print(time_day[time_day['Hour'].isin([7, 8])].to_string(index=False))

# ── 6. COVID EFFECT ───────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("6. COVID EFFECT — PRE / DURING / POST")
print(SEP)
df_nonzero = df_nonzero.copy()
df_nonzero['Era'] = pd.cut(
    df_nonzero['Year'],
    bins=[2014, 2019, 2021, 2025],
    labels=['Pre-COVID (2015-2019)', 'COVID (2020-2021)', 'Post-COVID (2022-2025)']
)
covid = (df_nonzero.groupby('Era', observed=True)
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Mins   = ('Min_Delay', 'mean'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
        Unique_Routes    = ('Route_Number', 'nunique'),
    )
    .round(1)
    .reset_index())
print(covid.to_string(index=False))

# ── 7. SEASONAL PATTERNS ─────────────────────────────────────────────────────
print(f"\n{SEP}")
print("7. SEASONAL PATTERNS")
print(SEP)
seasonal = (df_nonzero.groupby(['Season', 'Year'])
    .agg(
        Total_Incidents  = ('Min_Delay', 'count'),
        Avg_Delay_Mins   = ('Min_Delay', 'mean'),
        Severe_Rate_Pct  = ('Is_Severe', lambda x: x.mean() * 100),
    )
    .round(1)
    .reset_index()
    .sort_values(['Year', 'Season']))
print("Sample (2022-2025):")
print(seasonal[seasonal['Year'] >= 2022].to_string(index=False))

print(f"\n{SEP}")
print("PREVIEW COMPLETE — paste output to confirm, then we generate all tables")
print(SEP)