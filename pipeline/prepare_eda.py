import pandas as pd
import numpy as np
from datetime import datetime

INPUT_FILE  = "../data/processed/master_ttc_with_weather.csv"
OUTPUT_FILE = "../data/processed/master_ttc_eda_ready.csv"
LOG_FILE    = "../logs/eda_cleaning_log.txt"

log_lines = []
def log(text=""):
    print(text)
    log_lines.append(str(text))

log(f"EDA Cleaning started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv(INPUT_FILE, low_memory=False)
df['Service_DateTime'] = pd.to_datetime(df['Service_DateTime'], errors='coerce')
log(f"Loaded: {len(df):,} rows\n")

# ── 1. CAP EXTREME DELAYS ─────────────────────────────────────────────────────
log("[ 1. CAPPING EXTREME DELAYS ]")
log(f"  Rows with delay >= 300 min (5 hrs): {(df['Min_Delay'] >= 300).sum():,}")
log(f"  Rows with delay >= 999 min:         {(df['Min_Delay'] >= 999).sum():,}")

# 999+ are data entry caps or errors — cap at 300 min (5 hours, a realistic max)
# Anything genuinely over 5 hours would be an extraordinary event worth flagging
cap_value = 300
capped = (df['Min_Delay'] > cap_value).sum()
df.loc[df['Min_Delay'] > cap_value, 'Min_Delay'] = cap_value
log(f"  Capped {capped:,} rows to {cap_value} min  (values were {cap_value}+ min, clearly erroneous)")
log(f"  New max delay: {df['Min_Delay'].max()}")

# Same for Min_Gap
gap_capped = (df['Min_Gap'] > cap_value).sum()
df.loc[df['Min_Gap'] > cap_value, 'Min_Gap'] = cap_value
log(f"  Gap capped: {gap_capped:,} rows")

# ── 2. FILTER BAD ROUTE NUMBERS ───────────────────────────────────────────────
log(f"\n[ 2. FILTERING BAD ROUTE NUMBERS ]")
log(f"  Route range before: {df['Route_Number'].min():.0f} → {df['Route_Number'].max():.0f}")

# TTC bus routes are 1–999 (a few express routes go to ~985)
# Route 0 and anything > 999 are data entry errors
bad_routes = df['Route_Number'].notna() & ((df['Route_Number'] < 1) | (df['Route_Number'] > 999))
log(f"  Bad route rows (< 1 or > 999): {bad_routes.sum():,}")
log(f"  Sample bad route values: {sorted(df.loc[bad_routes, 'Route_Number'].unique()[:10])}")
df.loc[bad_routes, 'Route_Number'] = np.nan
log(f"  Route range after:  {df['Route_Number'].min():.0f} → {df['Route_Number'].max():.0f}")
log(f"  Total null routes now: {df['Route_Number'].isnull().sum():,}")

# ── 3. DERIVE EDA COLUMNS ─────────────────────────────────────────────────────
log(f"\n[ 3. DERIVING EDA COLUMNS ]")

df['Month']       = df['Service_DateTime'].dt.month
df['Hour']        = df['Service_DateTime'].dt.hour
df['Season']      = df['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3:  'Spring', 4: 'Spring', 5: 'Spring',
    6:  'Summer', 7: 'Summer', 8: 'Summer',
    9:  'Fall',  10: 'Fall',  11: 'Fall'
})
df['Is_Rush_Hour'] = df['Hour'].isin([7, 8, 9, 16, 17, 18])
df['Is_Weekend']   = df['Day'].isin(['Saturday', 'Sunday'])
df['Is_Severe']    = df['Min_Delay'] >= 15   # 15+ min = service impact threshold

log(f"  Month, Hour, Season, Is_Rush_Hour, Is_Weekend, Is_Severe  → added")
log(f"\n  Season distribution:")
for season, count in df['Season'].value_counts().items():
    log(f"    {season:<8}  {count:,}")

log(f"\n  Rush hour rows:   {df['Is_Rush_Hour'].sum():,}  ({df['Is_Rush_Hour'].mean()*100:.1f}%)")
log(f"  Weekend rows:     {df['Is_Weekend'].sum():,}  ({df['Is_Weekend'].mean()*100:.1f}%)")
log(f"  Severe delays:    {df['Is_Severe'].sum():,}  ({df['Is_Severe'].mean()*100:.1f}%)")

# ── 4. FINAL SHAPE & NULL SUMMARY ────────────────────────────────────────────
log(f"\n[ 4. FINAL DATASET SUMMARY ]")
log(f"  Shape: {df.shape[0]:,} rows  x  {df.shape[1]} columns")
log(f"  Columns: {list(df.columns)}")
log(f"\n  Null summary:")
for col in df.columns:
    n   = df[col].isnull().sum()
    pct = n / len(df) * 100
    if n > 0:
        log(f"    {col:<25} {n:>8,}  ({pct:.1f}%)")

log(f"\n  Delay stats after capping:")
log(f"    Min:    {df['Min_Delay'].min()}")
log(f"    Max:    {df['Min_Delay'].max()}")
log(f"    Mean:   {df['Min_Delay'].mean():.1f}")
log(f"    Median: {df['Min_Delay'].median():.1f}")

# ── 5. SAVE ───────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)
sz = __import__('os').path.getsize(OUTPUT_FILE) / 1024 / 1024
log(f"\n✅ Saved: {OUTPUT_FILE}  ({sz:.1f} MB)")

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"📋 Log saved: {LOG_FILE}")
log(f"\nEDA cleaning complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")