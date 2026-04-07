"""
geocode_nominatim_filtered.py — TTC Location Geocoding (filtered)
Only geocodes locations with frequency >= MIN_FREQUENCY (default 25).
Covers ~80% of all delay incidents in ~48 minutes.
Safe to interrupt and resume — progress saved every 50 results.
Run with: python geocode_nominatim_filtered.py  (from pipeline/ folder)
"""
import pandas as pd
import requests
import time
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE    = "../data/geocoding/locations_to_geocode.csv"
RESULTS_FILE  = "../data/geocoding/geocoded_locations.csv"
LOG_FILE      = "../logs/geocoding_log.txt"
MIN_FREQUENCY = 25
DELAY_SECONDS = 1.1
USER_AGENT    = "TTC-Delay-Research/1.0"
BATCH_SIZE    = 50
# ─────────────────────────────────────────────────────────────────────────────

log_lines = []

def log(text=""):
    print(text)
    log_lines.append(str(text))

def save_log():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

def geocode_one(query):
    url    = "https://nominatim.openstreetmap.org/search"
    params = {
        "q":            query,
        "format":       "json",
        "limit":        1,
        "countrycodes": "ca",
        "viewbox":      "-79.7,43.5,-79.1,43.9",
        "bounded":      1,
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        if results:
            return (
                float(results[0]['lat']),
                float(results[0]['lon']),
                results[0].get('display_name', '')
            )
        return None, None, "NO_RESULT"
    except Exception as e:
        return None, None, f"ERROR: {e}"

# ── LOAD AND FILTER ───────────────────────────────────────────────────────────
log(f"Geocoding started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

all_locs = pd.read_csv(INPUT_FILE)
log(f"Total unique locations in file: {len(all_locs):,}")

locs = all_locs[all_locs['Frequency'] >= MIN_FREQUENCY].copy()
log(f"Locations with frequency >= {MIN_FREQUENCY}: {len(locs):,}")

total_incidents   = all_locs['Frequency'].sum()
covered_incidents = locs['Frequency'].sum()
log(f"Incidents covered: {covered_incidents:,} / {total_incidents:,} ({covered_incidents/total_incidents*100:.1f}%)")
log(f"Estimated runtime: ~{len(locs) * DELAY_SECONDS / 60:.0f} minutes")

# ── RESUME FROM PREVIOUS RUN ──────────────────────────────────────────────────
if os.path.exists(RESULTS_FILE):
    done     = pd.read_csv(RESULTS_FILE)
    done_set = set(done['Location_Norm'].tolist())
    locs     = locs[~locs['Location_Norm'].isin(done_set)]
    log(f"Resuming — already done: {len(done_set):,}  |  remaining: {len(locs):,}")
else:
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    pd.DataFrame(columns=['Location_Norm', 'Frequency', 'Lat', 'Lon', 'Display_Name']
                 ).to_csv(RESULTS_FILE, index=False)
    log(f"Starting fresh — {len(locs):,} locations to process")

# ── GEOCODE LOOP ──────────────────────────────────────────────────────────────
total     = len(locs)
success   = 0
no_result = 0
errors    = 0
batch     = []

log(f"\nProgress saved every {BATCH_SIZE} results — safe to Ctrl+C and resume\n")

for i, (_, row) in enumerate(locs.iterrows(), 1):
    lat, lon, display = geocode_one(row['Search_Query'])

    batch.append({
        'Location_Norm': row['Location_Norm'],
        'Frequency':     row['Frequency'],
        'Lat':           lat,
        'Lon':           lon,
        'Display_Name':  display,
    })

    if lat is not None:
        success += 1
    elif display == "NO_RESULT":
        no_result += 1
    else:
        errors += 1

    if i % 100 == 0 or i == total:
        remaining = (total - i) * DELAY_SECONDS / 60
        log(f"  [{i:>5,}/{total:,}]  {i/total*100:.1f}%  |"
            f"  ✅ {success}  ❌ {no_result} no result  ⚠️ {errors} errors"
            f"  |  ~{remaining:.1f} min left")
        save_log()

    if len(batch) >= BATCH_SIZE:
        pd.DataFrame(batch).to_csv(RESULTS_FILE, mode='a', header=False, index=False)
        batch = []

    time.sleep(DELAY_SECONDS)

if batch:
    pd.DataFrame(batch).to_csv(RESULTS_FILE, mode='a', header=False, index=False)

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
final = pd.read_csv(RESULTS_FILE)
log(f"\n{'='*55}")
log(f"GEOCODING COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Total processed:  {len(final):,}")
log(f"  With coords:      {final['Lat'].notna().sum():,}  ({final['Lat'].notna().sum()/len(final)*100:.1f}%)")
log(f"  No result:        {(final['Display_Name']=='NO_RESULT').sum():,}")
log(f"  Errors:           {final['Display_Name'].str.startswith('ERROR').sum():,}")
log(f"\nResults saved to: {RESULTS_FILE}")
save_log()