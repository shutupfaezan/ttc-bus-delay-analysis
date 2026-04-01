import os
import pandas as pd
import numpy as np
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
BUS_DELAY_DIR = "../data/raw/Bus Delay Data"
OUTPUT_CSV    = "../data/processed/master_ttc_delays.csv"
LOG_FILE      = "../logs/master_ttc_build_log.txt"
# ─────────────────────────────────────────────────────────────────────────────

log_lines = []

def log(text=""):
    print(text)
    log_lines.append(str(text))

# ── DIRECTION HARMONIZER ──────────────────────────────────────────────────────
DIRECTION_MAP = {}
_north = ['N','n','NB','nb','n/b','N/B','Nb','N.B','n/B','North',"n/b'",'N/b','n/b ']
_south = ['S','s','SB','sb','s/b','S/B','Sb','s.b','s/B','SB.','S/b','s/b e/b']
_east  = ['E','e','EB','eb','e/b','E/B','Eb','E/b','east','e/B']
_west  = ['W','w','WB','wb','w/b','W/B','Wb','w/B','W/b','w/b ']
_both  = ['B','b','BW','bw','B/W','b/w','Bw','BN',"B/W's","b/w's",'b/w/B',
          'BWS','Both','Both ways','Both Ways','Both Way','Bothways','bothways',
          'BOTH','OB','ob','O/B','o/b','up','Up','UP','down','Down','dn',
          'BV','BW`','B.W','bw ']
for v in _north: DIRECTION_MAP[v] = 'N'
for v in _south: DIRECTION_MAP[v] = 'S'
for v in _east:  DIRECTION_MAP[v] = 'E'
for v in _west:  DIRECTION_MAP[v] = 'W'
for v in _both:  DIRECTION_MAP[v] = 'B'

def harmonize_direction(val):
    if pd.isnull(val) or str(val).strip() in ['nan', '', '-']:
        return np.nan
    v = str(val).strip()
    if v in DIRECTION_MAP:
        return DIRECTION_MAP[v]
    if v.upper() in ['N','S','E','W','B']:
        return v.upper()
    return np.nan

# ── NORMALIZERS ───────────────────────────────────────────────────────────────
def normalize_time(val):
    if pd.isnull(val): return np.nan
    if hasattr(val, 'strftime'): return val.strftime("%H:%M")
    s = str(val).strip()
    return s[:5] if (len(s) >= 5 and ':' in s) else np.nan

def normalize_date(val):
    try:
        ts = pd.to_datetime(val)
        return np.nan if pd.isnull(ts) else ts.strftime("%Y-%m-%d")
    except: return np.nan

def extract_route(val):
    if pd.isnull(val): return np.nan
    s = str(val).strip()
    try: return int(float(s))
    except ValueError: pass
    parts = s.split()
    return int(parts[0]) if (parts and parts[0].isdigit()) else np.nan

# ── SCHEMA RENAME (applied per-sheet before concat) ───────────────────────────
RENAME_MAP = {
    'Report Date': 'Service_Date', 'Date':      'Service_Date',
    'Route':       'Route_Number', 'Line':       'Route_Number',
    'Location':    'Location',     'Station':    'Location',
    'Incident':    'Incident_Code','Code':       'Incident_Code',
    'Delay':       'Min_Delay',    'Min Delay':  'Min_Delay',
    'Gap':         'Min_Gap',      'Min Gap':    'Min_Gap',
    'Direction':   'Direction',    'Bound':      'Direction',
}
JUNK_COLS  = {'_id', '_sheet', 'Incident ID'}
FINAL_COLS = ['Service_Date','Time','Route_Number','Day','Location',
              'Incident_Code','Min_Delay','Min_Gap','Direction',
              'Vehicle','Year','Source_File']

def normalize_sheet(df):
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in JUNK_COLS if c in df.columns])
    df = df.rename(columns=RENAME_MAP)
    return df.loc[:, ~df.columns.duplicated(keep='first')]

# ── LOAD ALL SHEETS FROM XLSX ─────────────────────────────────────────────────
def load_xlsx(fpath):
    xl, frames, skipped = pd.ExcelFile(fpath), [], []
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
            if df.empty or len(df.columns) < 3:
                skipped.append(sheet); continue
            df = normalize_sheet(df)
            df['_sheet'] = sheet
            frames.append(df)
        except Exception as e:
            skipped.append(f"{sheet}({e})")
    if skipped: log(f"    Skipped sheets: {skipped}")
    if not frames: raise ValueError("No valid sheets found.")
    combined = pd.concat(frames, ignore_index=True)
    log(f"    Sheets loaded: {len(frames)}/{len(xl.sheet_names)}  →  {len(combined):,} raw rows")
    return combined

# ── CLEAN ─────────────────────────────────────────────────────────────────────
def clean(df, fname):
    df = df.drop(columns=[c for c in ['_sheet'] if c in df.columns])
    df = normalize_sheet(df)                          # safety pass for CSVs

    df['Service_Date']  = df['Service_Date'].apply(normalize_date)
    df = df.dropna(subset=['Service_Date'])

    df['Time']          = df['Time'].apply(normalize_time)
    df['Route_Number']  = df['Route_Number'].apply(extract_route)
    df['Direction']     = df['Direction'].apply(harmonize_direction)
    df['Min_Delay']     = pd.to_numeric(df['Min_Delay'], errors='coerce')
    df['Min_Gap']       = pd.to_numeric(df['Min_Gap'],   errors='coerce')

    # Null out negative delays (data entry errors)
    df.loc[df['Min_Delay'] < 0, 'Min_Delay'] = np.nan

    df = df.drop_duplicates()
    df['Source_File'] = fname
    df['Year']        = pd.to_datetime(df['Service_Date']).dt.year
    return df[[c for c in FINAL_COLS if c in df.columns]]

# ── MAIN ──────────────────────────────────────────────────────────────────────
log(f"Build started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Source folder: {BUS_DELAY_DIR}\n")

files = sorted([f for f in os.listdir(BUS_DELAY_DIR)
                if f.endswith('.xlsx') or f.endswith('.csv')])
log(f"Files found: {len(files)}")
for f in files: log(f"  {f}")

all_frames = []

for fname in files:
    fpath = os.path.join(BUS_DELAY_DIR, fname)
    log(f"\n{'─'*65}")
    log(f"FILE: {fname}")

    try:
        if fname.endswith('.xlsx'):
            raw = load_xlsx(fpath)
        else:
            raw = pd.read_csv(fpath)
            log(f"    CSV rows loaded: {len(raw):,}")

        raw_count   = len(raw)
        cleaned     = clean(raw, fname)
        clean_count = len(cleaned)

        # Monthly breakdown
        monthly = (pd.to_datetime(cleaned['Service_Date'])
                   .dt.to_period('M').value_counts().sort_index())

        log(f"    Raw: {raw_count:,}  →  Clean: {clean_count:,}  (removed {raw_count - clean_count:,})")
        log(f"    Coverage: {monthly.index.min()} → {monthly.index.max()}  ({len(monthly)} months)")
        log(f"    Rows per month:")
        for period, count in monthly.items():
            log(f"      {period}:  {count:,}")

        # Column null summary for this file
        log(f"    Nulls per column:")
        for col in cleaned.columns:
            n = cleaned[col].isnull().sum()
            if n > 0:
                log(f"      {col:<20} {n:>6,}  ({n/clean_count*100:.1f}%)")

        all_frames.append(cleaned)

    except Exception as e:
        import traceback
        log(f"  ❌ ERROR: {e}")
        log(traceback.format_exc())

# ── STACK & SAVE ──────────────────────────────────────────────────────────────
log(f"\n{'═'*65}")
log("STACKING ALL FILES")

master = pd.concat(all_frames, ignore_index=True)

master['Service_DateTime'] = pd.to_datetime(
    master['Service_Date'] + ' ' + master['Time'].fillna('00:00'),
    errors='coerce'
)

# ── FINAL REPORT ──────────────────────────────────────────────────────────────
log(f"\n{'═'*65}")
log(f"MASTER DATASET — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"{'═'*65}")
log(f"  Total rows:    {len(master):,}")
log(f"  Total columns: {len(master.columns)}")

log(f"\n  ROWS PER YEAR:")
for year, count in master['Year'].value_counts().sort_index().items():
    log(f"    {year}:  {count:,}")

log(f"\n  ROWS PER MONTH (all years):")
all_monthly = (pd.to_datetime(master['Service_Date'])
               .dt.to_period('M').value_counts().sort_index())
for period, count in all_monthly.items():
    log(f"    {period}:  {count:,}")

log(f"\n  NULL SUMMARY:")
for col in master.columns:
    n   = master[col].isnull().sum()
    pct = n / len(master) * 100
    flag = "  ⚠️  review" if pct > 5 else ""
    log(f"    {col:<20} {n:>8,}  ({pct:.1f}%){flag}")

master.to_csv(OUTPUT_CSV, index=False)
log(f"\n✅ CSV saved: {OUTPUT_CSV}  ({os.path.getsize(OUTPUT_CSV)/1024/1024:.1f} MB)")

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"📋 Log saved: {LOG_FILE}")
log(f"\nBuild complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")