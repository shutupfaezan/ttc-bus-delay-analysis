"""
prepare_model_v2.py — TTC Bus Delay Regression
Prepares master_ttc_eda_ready.csv for regression modelling.
Target: Min_Delay (continuous, minutes behind schedule)
Run with: python prepare_model_v2.py
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "../data/processed/master_ttc_eda_ready.csv"
OUTPUT_FILE = "../data/processed/master_ttc_model_ready.csv"
LOG_FILE    = "../logs/prepare_model_v2_log.txt"
# ─────────────────────────────────────────────────────────────────────────────

log_lines = []
def log(text=""):
    print(text)
    log_lines.append(str(text))

log(f"prepare_model_v2 started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Target: Min_Delay (regression — minutes behind schedule)")

# ── LOAD ──────────────────────────────────────────────────────────────────────
log("\n[ LOADING ]")
df = pd.read_csv(INPUT_FILE, low_memory=False)
log(f"  Rows loaded:  {len(df):,}")
log(f"  Columns:      {len(df.columns)}")

# ── STEP 1 — DROP ZERO DELAY ROWS ────────────────────────────────────────────
# Zero-delay rows mean no incident occurred — nothing to predict
log("\n[ STEP 1 — DROP ZERO AND NULL DELAY ROWS ]")
before = len(df)
df = df[df['Min_Delay'] > 0].copy()
df = df[df['Min_Delay'].notna()].copy()
log(f"  Rows removed (Min_Delay = 0 or null): {before - len(df):,}")
log(f"  Rows remaining:                       {len(df):,}")

# ── STEP 2 — INCIDENT CODE UNIFIED MAPPING ───────────────────────────────────
log("\n[ STEP 2 — INCIDENT CODE UNIFICATION ]")

INCIDENT_MAP = {
    # ── MECHANICAL ────────────────────────────────────────────────────────────
    'Mechanical':                        'Mechanical',
    'MFO':                               'Mechanical',
    'MFSH':                              'Mechanical',
    'MFS':                               'Mechanical',
    'MFFD':                              'Mechanical',
    'MFPR':                              'Mechanical',
    'MFWEA':                             'Mechanical',
    'MFLD':                              'Mechanical',
    'Late Entering Service - Mechanical':'Mechanical',
    'MTO':                               'Mechanical',
    'MTUS':                              'Mechanical',
    'MTIE':                              'Mechanical',

    # ── DIVERSION ─────────────────────────────────────────────────────────────
    'Diversion':                         'Diversion',
    'MFDV':                              'Diversion',
    'EFD':                               'Diversion',
    'MTDV':                              'Diversion',

    # ── UTILIZED OFF ROUTE ────────────────────────────────────────────────────
    'Utilized Off Route':                'Utilized Off Route',
    'Utilizing Off Route':               'Utilized Off Route',
    'MFTO':                              'Utilized Off Route',
    'MFUI':                              'Utilized Off Route',
    'MFUIR':                             'Utilized Off Route',
    'MUO':                               'Utilized Off Route',
    'MUIE':                              'Utilized Off Route',
    'MUIS':                              'Utilized Off Route',

    # ── TRAFFIC ───────────────────────────────────────────────────────────────
    'TFO':                               'Traffic',
    'TFCNO':                             'Traffic',
    'TFPD':                              'Traffic',
    'TFOI':                              'Traffic',
    'TFLF':                              'Traffic',
    'TFLL':                              'Traffic',
    'TFPI':                              'Traffic',

    # ── GENERAL DELAY ─────────────────────────────────────────────────────────
    'General Delay':                     'General Delay',
    'Held By':                           'General Delay',
    'Late':                              'General Delay',
    'Operations':                        'General Delay',
    'SUDP':                              'General Delay',

    # ── LATE DEPARTURE ────────────────────────────────────────────────────────
    'Late Leaving Garage':               'Late Departure',
    'Late Leaving Garage - Mechanical':  'Late Departure',
    'Late Leaving Garage - Operator':    'Late Departure',
    'Late Leaving Garage - Management':  'Late Departure',
    'Late Leaving Garage - Vision':      'Late Departure',
    'Late Leaving Garage - Operations':  'Late Departure',
    'Late Entering Service':             'Late Departure',
    'PREL':                              'Late Departure',
    'MFESA':                             'Late Departure',

    # ── INVESTIGATION ─────────────────────────────────────────────────────────
    'Investigation':                     'Investigation',

    # ── EMERGENCY SERVICES ────────────────────────────────────────────────────
    'Emergency Services':                'Emergency Services',
    'EFO':                               'Emergency Services',
    'EFB':                               'Emergency Services',
    'EFHVA':                             'Emergency Services',
    'EFRA':                              'Emergency Services',
    'EFCAN':                             'Emergency Services',
    'ETO':                               'Emergency Services',

    # ── SECURITY ──────────────────────────────────────────────────────────────
    'Security':                          'Security',
    'SFDP':                              'Security',
    'SFO':                               'Security',
    'SFPOL':                             'Security',
    'SFAP':                              'Security',
    'SFAE':                              'Security',
    'SFSA':                              'Security',
    'SFSP':                              'Security',
    'STO':                               'Security',
    'SRO':                               'Security',
    'Securitty':                         'Security',

    # ── COLLISION ─────────────────────────────────────────────────────────────
    'Collision - TTC':                   'Collision',
    'Collision - TTC Involved':          'Collision',
    'Road Blocked - NON-TTC Collision':  'Collision',
    'Road Block - Non-TTC Collision':    'Collision',
    'Roadblock by Collision - Non-TTC':  'Collision',
    'TTPD':                              'Collision',

    # ── OPERATOR ──────────────────────────────────────────────────────────────
    'Operations - Operator':             'Operator',
    'Management':                        'Operator',

    # ── CLEANING ──────────────────────────────────────────────────────────────
    'Cleaning':                          'Cleaning',
    'Cleaning - Unsanitary':             'Cleaning',
    'Cleaning - Disinfection':           'Cleaning',
    'MFSAN':                             'Cleaning',
    'MFUS':                              'Cleaning',
    'MTSAN':                             'Cleaning',

    # ── VISION ────────────────────────────────────────────────────────────────
    'Vision':                            'Vision',
    'MFVIS':                             'Vision',
    'MTVIS':                             'Vision',

    # ── PASSENGER ─────────────────────────────────────────────────────────────
    'PFO':                               'Passenger',
    'PFPD':                              'Passenger',
    'EFP':                               'Passenger',
    'MFPI':                              'Passenger',

    # ── OTHER ─────────────────────────────────────────────────────────────────
    'Overhead':                          'Other',
    'Rail/Switches':                     'Other',
    'e':                                 'Other',
    'MTNOA':                             'Other',
}

df['Incident_Category'] = df['Incident_Code'].map(INCIDENT_MAP)

mapped   = df['Incident_Category'].notna().sum()
unmapped = df['Incident_Category'].isna().sum()
log(f"  Mapped:   {mapped:,} rows")
log(f"  Unmapped: {unmapped:,} rows (will be dropped — no category = unusable for ops model)")

# Drop rows with no incident category — dispatcher always knows the incident type
before = len(df)
df = df[df['Incident_Category'].notna()].copy()
log(f"  Rows dropped (unmapped incident):  {before - len(df):,}")
log(f"  Rows remaining:                    {len(df):,}")

log(f"\n  Category distribution:")
cat_counts = df['Incident_Category'].value_counts()
for cat, count in cat_counts.items():
    avg_delay = df[df['Incident_Category'] == cat]['Min_Delay'].mean()
    log(f"    {cat:<25} {count:>8,}  avg delay: {avg_delay:.1f} min")

# ── STEP 3 — DROP ROWS WITH NULL FEATURES ────────────────────────────────────
log("\n[ STEP 3 — DROP ROWS WITH NULL FEATURES ]")
required_cols = ['Route_Number', 'Hour', 'Month', 'Day', 'Season',
                 'Is_Rush_Hour', 'Is_Weekend',
                 'Temp_C', 'Visibility_km', 'Wind_Spd_kmh', 'Rel_Humidity_pct']
before = len(df)
df = df.dropna(subset=required_cols).copy()
log(f"  Rows dropped (null features): {before - len(df):,}")
log(f"  Rows remaining:               {len(df):,}")

# ── STEP 4 — SELECT AND ORDER COLUMNS ────────────────────────────────────────
log("\n[ STEP 4 — SELECT MODEL COLUMNS ]")

KEEP_COLS = [
    # Target
    'Min_Delay',
    # Route & time
    'Route_Number',
    'Hour',
    'Month',
    'Day',
    'Season',
    'Is_Rush_Hour',
    'Is_Weekend',
    # Weather
    'Temp_C',
    'Visibility_km',
    'Wind_Spd_kmh',
    'Rel_Humidity_pct',
    # Incident
    'Incident_Category',
    # Reference only — not used as features
    'Year',
    'Service_Date',
]

model_df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
log(f"  Columns selected: {list(model_df.columns)}")

# ── STEP 5 — ENCODE BOOLEANS ──────────────────────────────────────────────────
log("\n[ STEP 5 — ENCODE BOOLEANS ]")
for col in ['Is_Rush_Hour', 'Is_Weekend']:
    model_df[col] = model_df[col].astype(int)
    log(f"  {col} → 0/1 integer")

# ── STEP 6 — TARGET SUMMARY ───────────────────────────────────────────────────
log("\n[ STEP 6 — TARGET VARIABLE SUMMARY ]")
log(f"  Target: Min_Delay (minutes behind schedule)")
log(f"  Min:    {model_df['Min_Delay'].min():.0f} min")
log(f"  Max:    {model_df['Min_Delay'].max():.0f} min")
log(f"  Mean:   {model_df['Min_Delay'].mean():.1f} min")
log(f"  Median: {model_df['Min_Delay'].median():.1f} min")
log(f"  Std:    {model_df['Min_Delay'].std():.1f} min")
log(f"\n  Distribution buckets:")
buckets = [0, 5, 10, 15, 30, 60, 300]
for i in range(len(buckets) - 1):
    lo, hi = buckets[i], buckets[i+1]
    n   = ((model_df['Min_Delay'] > lo) & (model_df['Min_Delay'] <= hi)).sum()
    pct = n / len(model_df) * 100
    log(f"    {lo:>3}–{hi:<3} min:  {n:>8,}  ({pct:.1f}%)")

log(f"\n  Year split preview (for time-based train/val/test):")
for year, grp in model_df.groupby('Year'):
    log(f"    {year}:  {len(grp):>7,} rows  avg delay: {grp['Min_Delay'].mean():.1f} min")

# ── SAVE ──────────────────────────────────────────────────────────────────────
log("\n[ SAVING ]")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
model_df.to_csv(OUTPUT_FILE, index=False)
sz = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
log(f"  Saved: {OUTPUT_FILE}  ({sz:.1f} MB)")
log(f"  Rows:    {len(model_df):,}")
log(f"  Columns: {len(model_df.columns)}")

log(f"\n{'='*60}")
log("READY FOR TRAINING")
log(f"{'='*60}")
log(f"  Target:    Min_Delay (continuous regression)")
log(f"  Train:     Year <= 2023")
log(f"  Validate:  Year == 2024")
log(f"  Test:      Year == 2025")
log(f"  Next step: run train_model_v3.py")
log(f"\nprepare_model_v2 complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Log saved: {LOG_FILE}")