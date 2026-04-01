import pandas as pd
import numpy as np
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
TTC_FILE     = "../data/processed/master_ttc_delays.csv"
WEATHER_FILE = "../data/raw/Weather Data/master_weather_data.csv"
OUTPUT_FILE  = "../data/processed/master_ttc_with_weather.csv"
LOG_FILE     = "../logs/join_weather_log.txt"
# ─────────────────────────────────────────────────────────────────────────────

log_lines = []
def log(text=""):
    print(text)
    log_lines.append(str(text))

log(f"Weather join started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── LOAD TTC ──────────────────────────────────────────────────────────────────
log("\n[ LOADING TTC MASTER ]")
ttc = pd.read_csv(TTC_FILE)
ttc['Service_DateTime'] = pd.to_datetime(ttc['Service_DateTime'], errors='coerce')
log(f"  TTC rows: {len(ttc):,}")
log(f"  DateTime nulls: {ttc['Service_DateTime'].isnull().sum()}")

# ── LOAD & CLEAN WEATHER ──────────────────────────────────────────────────────
log("\n[ LOADING WEATHER ]")
wdf = pd.read_csv(WEATHER_FILE, low_memory=False)
log(f"  Raw weather rows: {len(wdf):,}")

# Drop all Flag columns (100% null) and other useless columns
drop_cols = [c for c in wdf.columns if 'Flag' in c or 'flag' in c]
drop_cols += ['Longitude (x)', 'Latitude (y)', 'Station Name', 'Climate ID',
              'Year', 'Month', 'Day', 'Time (LST)',   # redundant with Date/Time
              'Hmdx',                                  # 82.5% null
              'Precip. Amount (mm)']                   # 100% null
drop_cols = [c for c in drop_cols if c in wdf.columns]
wdf = wdf.drop(columns=drop_cols)
log(f"  Dropped {len(drop_cols)} columns (flags, redundant, high-null)")

# Rename for clarity
wdf = wdf.rename(columns={
    'Date/Time (LST)':      'Weather_DateTime',
    'Temp (°C)':            'Temp_C',
    'Dew Point Temp (°C)':  'Dew_Point_C',
    'Rel Hum (%)':          'Rel_Humidity_pct',
    'Wind Dir (10s deg)':   'Wind_Dir_10deg',
    'Wind Spd (km/h)':      'Wind_Spd_kmh',
    'Visibility (km)':      'Visibility_km',
    'Stn Press (kPa)':      'Stn_Pressure_kPa',
    'Wind Chill':           'Wind_Chill',
    'Weather':              'Weather_Desc',
})

# Parse weather datetime and floor to hour for join key
wdf['Weather_DateTime'] = pd.to_datetime(wdf['Weather_DateTime'], errors='coerce')
wdf = wdf.dropna(subset=['Weather_DateTime'])
log(f"  Clean weather rows: {len(wdf):,}")
log(f"  Weather range: {wdf['Weather_DateTime'].min()} → {wdf['Weather_DateTime'].max()}")

# ── CREATE JOIN KEYS (floor both to nearest hour) ─────────────────────────────
log("\n[ CREATING JOIN KEYS ]")
ttc['Join_Hour'] = ttc['Service_DateTime'].dt.floor('h')
wdf['Join_Hour'] = wdf['Weather_DateTime'].dt.floor('h')

log(f"  TTC unique hours:     {ttc['Join_Hour'].nunique():,}")
log(f"  Weather unique hours: {wdf['Join_Hour'].nunique():,}")

# ── MERGE ─────────────────────────────────────────────────────────────────────
log("\n[ MERGING ]")
weather_cols = ['Join_Hour','Temp_C','Dew_Point_C','Rel_Humidity_pct',
                'Wind_Dir_10deg','Wind_Spd_kmh','Visibility_km',
                'Stn_Pressure_kPa','Wind_Chill','Weather_Desc']

merged = ttc.merge(
    wdf[weather_cols],
    on='Join_Hour',
    how='left'   # keep all TTC rows, attach weather where available
)

# Drop the helper column
merged = merged.drop(columns=['Join_Hour'])

log(f"  TTC rows before: {len(ttc):,}")
log(f"  TTC rows after:  {len(merged):,}")

if len(merged) != len(ttc):
    log(f"  ⚠️  Row count changed — possible duplicate join hours in weather data!")
    # Fix: deduplicate weather on Join_Hour before merge
    log(f"  Deduplicating weather on Join_Hour and re-merging...")
    wdf_deduped = wdf[weather_cols].drop_duplicates(subset=['Join_Hour'])
    merged = ttc.merge(wdf_deduped, on='Join_Hour', how='left')
    merged = merged.drop(columns=['Join_Hour'])
    log(f"  Rows after dedup merge: {len(merged):,}")

# ── REPORT ────────────────────────────────────────────────────────────────────
log("\n[ JOIN QUALITY REPORT ]")
weather_feature_cols = ['Temp_C','Dew_Point_C','Rel_Humidity_pct',
                        'Wind_Dir_10deg','Wind_Spd_kmh','Visibility_km',
                        'Stn_Pressure_kPa','Wind_Chill','Weather_Desc']

for col in weather_feature_cols:
    if col in merged.columns:
        n   = merged[col].isnull().sum()
        pct = n / len(merged) * 100
        flag = "  ⚠️" if pct > 5 else ""
        log(f"  {col:<25} nulls: {n:>8,}  ({pct:.1f}%){flag}")

log(f"\n  Total rows:    {len(merged):,}")
log(f"  Total columns: {len(merged.columns)}")
log(f"  Columns: {list(merged.columns)}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
merged.to_csv(OUTPUT_FILE, index=False)
sz = __import__('os').path.getsize(OUTPUT_FILE) / 1024 / 1024
log(f"\n✅ Saved: {OUTPUT_FILE}  ({sz:.1f} MB)")

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"📋 Log saved: {LOG_FILE}")
log(f"\nWeather join complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")