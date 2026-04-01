import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "../data/processed/master_ttc_eda_ready.csv"
OUTPUT_FILE = "../data/processed/master_ttc_model_ready.csv"
LOG_FILE    = "../logs/prepare_model_log.txt"
# ─────────────────────────────────────────────────────────────────────────────

log_lines = []
def log(text=""):
    print(text)
    log_lines.append(str(text))

log(f"Model prep started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── LOAD ──────────────────────────────────────────────────────────────────────
log("\n[ LOADING ]")
df = pd.read_csv(INPUT_FILE, low_memory=False)
log(f"  Rows loaded: {len(df):,}")
log(f"  Columns: {len(df.columns)}")

# ── STEP 1 — FIX EMPTY STRINGS ───────────────────────────────────────────────
log("\n[ STEP 1 — FIX EMPTY STRINGS ]")
for col in ['Weather_Desc', 'Direction', 'Incident_Code', 'Location']:
    before = (df[col] == '').sum()
    df[col] = df[col].replace('', np.nan)
    log(f"  {col:<20} empty strings fixed: {before:,}")

# ── STEP 2 — REMOVE ZERO DELAY ROWS ─────────────────────────────────────────
log("\n[ STEP 2 — REMOVE ZERO DELAY ROWS ]")
before = len(df)
df = df[df['Min_Delay'] > 0].copy()
log(f"  Rows removed (Min_Delay = 0): {before - len(df):,}")
log(f"  Rows remaining: {len(df):,}")

# ── STEP 3 — INCIDENT CODE UNIFIED MAPPING ───────────────────────────────────
log("\n[ STEP 3 — INCIDENT CODE UNIFICATION ]")

INCIDENT_MAP = {

    # ── MECHANICAL ────────────────────────────────────────────────────────────
    # Bus hardware failures unrelated to diversion or vision
    'Mechanical':                        'Mechanical',
    'MFO':                               'Mechanical',   # Mechanical Failure - Other
    'MFSH':                              'Mechanical',   # Mechanical Failure - Shift
    'MFS':                               'Mechanical',   # Mechanical Failure - Shift (alt)
    'MFFD':                              'Mechanical',   # Mechanical Failure - Fluid Defect
    'MFPR':                              'Mechanical',   # Mechanical Failure - Personal Responsibility
    'MFWEA':                             'Mechanical',   # Mechanical Failure - Weather related
    'MFLD':                              'Mechanical',   # Mechanical Failure - Late Departure
    'Late Entering Service - Mechanical':'Mechanical',
    'MTO':                               'Mechanical',   # Mechanical Transfer - Other
    'MTUS':                              'Mechanical',   # Mechanical Transfer - Unsanitary
    'MTIE':                              'Mechanical',   # Mechanical Transfer - Incident Equipment

    # ── DIVERSION ─────────────────────────────────────────────────────────────
    # Bus taken off its scheduled route
    'Diversion':                         'Diversion',
    'MFDV':                              'Diversion',    # Mechanical Failure - Diversion
    'EFD':                               'Diversion',    # External Factor - Diversion
    'MTDV':                              'Diversion',    # Mechanical Transfer - Diversion

    # ── UTILIZED OFF ROUTE ────────────────────────────────────────────────────
    # Bus operating off route (distinct from forced diversion)
    'Utilized Off Route':                'Utilized Off Route',
    'Utilizing Off Route':               'Utilized Off Route',  # variant spelling
    'MFTO':                              'Utilized Off Route',  # Mechanical Failure - Traffic Other
    'MFUI':                              'Utilized Off Route',  # Mechanical Failure - Utilized
    'MFUIR':                             'Utilized Off Route',  # Mechanical Failure - Utilized Irregular
    'MUO':                               'Utilized Off Route',  # Mechanical - Utilized Off Route - Other
    'MUIE':                              'Utilized Off Route',  # Mechanical - Utilized Irregular Equipment
    'MUIS':                              'Utilized Off Route',  # Mechanical - Utilized Irregular Service

    # ── TRAFFIC ───────────────────────────────────────────────────────────────
    # External traffic conditions blocking or delaying the route
    'TFO':                               'Traffic',      # Traffic Factor - Other
    'TFCNO':                             'Traffic',      # Traffic Factor - Collision Non-TTC
    'TFPD':                              'Traffic',      # Traffic Factor - Pedestrian
    'TFOI':                              'Traffic',      # Traffic Factor - Obstruction Incident
    'TFLF':                              'Traffic',      # Traffic Factor - Left Fork
    'TFLL':                              'Traffic',      # Traffic Factor - Lane Loss
    'TFPI':                              'Traffic',      # Traffic Factor - Personal Injury

    # ── GENERAL DELAY ─────────────────────────────────────────────────────────
    # Non-specific or catch-all delay codes
    'General Delay':                     'General Delay',
    'Held By':                           'General Delay',
    'Late':                              'General Delay',
    'Operations':                        'General Delay',
    'SUDP':                              'General Delay',  # Service - Unplanned Departure

    # ── LATE DEPARTURE ────────────────────────────────────────────────────────
    # Bus left garage or entered service late
    'Late Leaving Garage':               'Late Departure',
    'Late Leaving Garage - Mechanical':  'Late Departure',
    'Late Leaving Garage - Operator':    'Late Departure',
    'Late Leaving Garage - Management':  'Late Departure',
    'Late Leaving Garage - Vision':      'Late Departure',
    'Late Leaving Garage - Operations':  'Late Departure',
    'Late Entering Service':             'Late Departure',
    'PREL':                              'Late Departure',  # Passenger Related - Early Leaving
    'MFESA':                             'Late Departure',  # Mechanical Failure - Emergency Services Attending (delays departure)

    # ── INVESTIGATION ─────────────────────────────────────────────────────────
    'Investigation':                     'Investigation',

    # ── EMERGENCY SERVICES ────────────────────────────────────────────────────
    # Police, fire, ambulance activity near or involving the bus
    'Emergency Services':                'Emergency Services',
    'EFO':                               'Emergency Services',  # External Factor - Other (emergency)
    'EFB':                               'Emergency Services',  # External Factor - Blocked
    'EFHVA':                             'Emergency Services',  # External Factor - High Visibility Area
    'EFRA':                              'Emergency Services',  # External Factor - Road/Area blockage
    'EFCAN':                             'Emergency Services',  # External Factor - Cancelled
    'ETO':                               'Emergency Services',  # External - Transfer Other

    # ── SECURITY ──────────────────────────────────────────────────────────────
    # Security incidents on or near the vehicle
    'Security':                          'Security',
    'SFDP':                              'Security',     # Security Factor - Disorderly Person
    'SFO':                               'Security',     # Security Factor - Other
    'SFPOL':                             'Security',     # Security Factor - Police
    'SFAP':                              'Security',     # Security Factor - Assault on Passenger
    'SFAE':                              'Security',     # Security Factor - Assault on Employee
    'SFSA':                              'Security',     # Security Factor - Suspicious Activity
    'SFSP':                              'Security',     # Security Factor - Suspicious Package
    'STO':                               'Security',     # Security - Transfer Other
    'SRO':                               'Security',     # Security - Route Other
    'Securitty':                         'Security',     # typo — map to Security

    # ── COLLISION ─────────────────────────────────────────────────────────────
    # Any vehicle collision involving or adjacent to TTC bus
    'Collision - TTC':                   'Collision',
    'Collision - TTC Involved':          'Collision',
    'Road Blocked - NON-TTC Collision':  'Collision',
    'Road Block - Non-TTC Collision':    'Collision',
    'Roadblock by Collision - Non-TTC':  'Collision',
    'TTPD':                              'Collision',    # TTC Personnel - Pedestrian

    # ── OPERATOR ──────────────────────────────────────────────────────────────
    # Delay caused by operator conduct or management decision
    'Operations - Operator':             'Operator',
    'Management':                        'Operator',

    # ── CLEANING ──────────────────────────────────────────────────────────────
    # Vehicle required cleaning before continuing service
    'Cleaning':                          'Cleaning',
    'Cleaning - Unsanitary':             'Cleaning',
    'Cleaning - Disinfection':           'Cleaning',
    'MFSAN':                             'Cleaning',     # Mechanical Failure - Sanitary
    'MFUS':                              'Cleaning',     # Mechanical Failure - Unsanitary
    'MTSAN':                             'Cleaning',     # Mechanical Transfer - Sanitary

    # ── VISION ────────────────────────────────────────────────────────────────
    # Operator vision obstruction (sun glare, dirty windshield, etc.)
    'Vision':                            'Vision',
    'MFVIS':                             'Vision',       # Mechanical Failure - Vision
    'MTVIS':                             'Vision',       # Mechanical Transfer - Vision

    # ── PASSENGER ─────────────────────────────────────────────────────────────
    # Passenger-related incidents causing delay
    'PFO':                               'Passenger',    # Passenger Factor - Other
    'PFPD':                              'Passenger',    # Passenger Factor - Passenger Disturbance
    'EFP':                               'Passenger',    # External Factor - Pedestrian
    'MFPI':                              'Passenger',    # Mechanical Failure - Personal Injury

    # ── OTHER ─────────────────────────────────────────────────────────────────
    # Rare, ambiguous, or unclassifiable codes
    'Overhead':                          'Other',        # Overhead wire issue (unusual for bus)
    'Rail/Switches':                     'Other',        # Rail issue (unusual for bus)
    'e':                                 'Other',        # Single character data entry error
    'MTNOA':                             'Other',        # Mechanical Transfer - No Operator Available
}

# Apply mapping
df['Incident_Category'] = df['Incident_Code'].map(INCIDENT_MAP)

# Report
mapped   = df['Incident_Category'].notna().sum()
unmapped = df['Incident_Category'].isna().sum()
log(f"  Mapped:   {mapped:,} rows")
log(f"  Unmapped: {unmapped:,} rows (will be null — check below)")

# Show any unmapped codes
if unmapped > 0:
    missing = df[df['Incident_Category'].isna()]['Incident_Code'].value_counts()
    log(f"  Unmapped codes:")
    for code, count in missing.items():
        log(f"    '{code}': {count:,} rows")

# Category distribution
log(f"\n  Category distribution:")
cat_counts = df['Incident_Category'].value_counts()
for cat, count in cat_counts.items():
    pct = count / len(df) * 100
    log(f"    {cat:<25} {count:>8,}  ({pct:.1f}%)")

# ── STEP 4 — SELECT MODEL FEATURES ───────────────────────────────────────────
log("\n[ STEP 4 — SELECT MODEL FEATURES ]")

FEATURE_COLS = [
    # Target
    'Is_Severe',
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
    # Keep for reference (not used as features)
    'Min_Delay',
    'Year',
    'Service_Date',
]

model_df = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
log(f"  Columns selected: {list(model_df.columns)}")

# ── STEP 5 — HANDLE REMAINING NULLS ──────────────────────────────────────────
log("\n[ STEP 5 — NULL SUMMARY BEFORE SAVE ]")
for col in model_df.columns:
    n = model_df[col].isna().sum()
    pct = n / len(model_df) * 100
    if n > 0:
        log(f"  {col:<25} nulls: {n:>8,}  ({pct:.1f}%)")

# ── STEP 6 — ENCODE BOOLEANS ─────────────────────────────────────────────────
log("\n[ STEP 6 — ENCODE BOOLEANS ]")
for col in ['Is_Rush_Hour', 'Is_Weekend', 'Is_Severe']:
    if col in model_df.columns:
        model_df[col] = model_df[col].astype(int)
        log(f"  {col} → 0/1 integer")

# ── SAVE ──────────────────────────────────────────────────────────────────────
log("\n[ SAVING ]")
model_df.to_csv(OUTPUT_FILE, index=False)
sz = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
log(f"  Saved: {OUTPUT_FILE}  ({sz:.1f} MB)")
log(f"  Rows: {len(model_df):,}")
log(f"  Columns: {len(model_df.columns)}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
log(f"\n{'='*60}")
log("READY FOR MODELLING")
log(f"{'='*60}")
log(f"  Target:   Is_Severe  (0 = not severe, 1 = severe)")
log(f"  Positive: {model_df['Is_Severe'].sum():,} rows ({model_df['Is_Severe'].mean()*100:.1f}%)")
log(f"  Negative: {(model_df['Is_Severe']==0).sum():,} rows ({(model_df['Is_Severe']==0).mean()*100:.1f}%)")
log(f"\n  Next step: run train_model.py")
log(f"\nModel prep complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Log saved: {LOG_FILE}")