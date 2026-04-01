import pandas as pd
import requests
import time
import json
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE    = "locations_to_geocode.csv"
RESULTS_FILE  = "geocoded_locations.csv"     # saves progress incrementally
LOG_FILE      = "geocoding_log.txt"
DELAY_SECONDS = 1.1                          # Nominatim requires 1 req/sec max
USER_AGENT    = "TTC-Delay-Research/1.0"     # required by Nominatim ToS
# ─────────────────────────────────────────────────────────────────────────────

log_lines = []
def log(text=""):
    print(text)
    log_lines.append(str(text))

def save_log():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

def geocode_one(query):
    """Query Nominatim for a single location string."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q":              query,
        "format":         "json",
        "limit":          1,
        "countrycodes":   "ca",              # Canada only — reduces false matches
        "viewbox":        "-79.7,43.5,-79.1,43.9",  # Toronto bounding box
        "bounded":        1,                 # stay within bounding box
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        if results:
            return float(results[0]['lat']), float(results[0]['lon']), results[0].get('display_name','')
        return None, None, "NO_RESULT"
    except Exception as e:
        return None, None, f"ERROR: {e}"

# ── LOAD LOCATIONS ────────────────────────────────────────────────────────────
locs = pd.read_csv(INPUT_FILE)
log(f"Geocoding started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Total unique locations to geocode: {len(locs):,}")

# ── RESUME FROM PREVIOUS RUN ──────────────────────────────────────────────────
if os.path.exists(RESULTS_FILE):
    done = pd.read_csv(RESULTS_FILE)
    done_set = set(done['Location_Norm'].tolist())
    locs = locs[~locs['Location_Norm'].isin(done_set)]
    log(f"Resuming — already done: {len(done_set):,}  |  remaining: {len(locs):,}")
else:
    # Create empty results file with headers
    pd.DataFrame(columns=['Location_Norm','Frequency','Lat','Lon','Display_Name']
                 ).to_csv(RESULTS_FILE, index=False)
    log(f"Starting fresh — {len(locs):,} locations to process")

# ── GEOCODE LOOP ──────────────────────────────────────────────────────────────
total      = len(locs)
success    = 0
no_result  = 0
errors     = 0
batch      = []
BATCH_SIZE = 50    # write to disk every 50 results

log(f"\nEstimated time: ~{total * DELAY_SECONDS / 3600:.1f} hours")
log(f"Progress will save every {BATCH_SIZE} results — safe to Ctrl+C and resume\n")

for i, (_, row) in enumerate(locs.iterrows(), 1):
    query = row['Search_Query']
    lat, lon, display = geocode_one(query)

    result = {
        'Location_Norm': row['Location_Norm'],
        'Frequency':     row['Frequency'],
        'Lat':           lat,
        'Lon':           lon,
        'Display_Name':  display,
    }
    batch.append(result)

    if lat is not None:
        success += 1
    elif display == "NO_RESULT":
        no_result += 1
    else:
        errors += 1

    # Progress update every 100
    if i % 100 == 0 or i == total:
        pct      = i / total * 100
        elapsed  = i * DELAY_SECONDS / 3600
        remaining = (total - i) * DELAY_SECONDS / 3600
        log(f"  [{i:>6,}/{total:,}]  {pct:.1f}%  |  ✅ {success}  ❌ {no_result} no result  ⚠️ {errors} errors  |  ~{remaining:.1f}h left")
        save_log()

    # Write batch to disk
    if len(batch) >= BATCH_SIZE:
        pd.DataFrame(batch).to_csv(RESULTS_FILE, mode='a', header=False, index=False)
        batch = []

    time.sleep(DELAY_SECONDS)

# Write any remaining
if batch:
    pd.DataFrame(batch).to_csv(RESULTS_FILE, mode='a', header=False, index=False)

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
final = pd.read_csv(RESULTS_FILE)
log(f"\n{'='*55}")
log(f"GEOCODING COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Total processed:  {len(final):,}")
log(f"  ✅ With coords:   {final['Lat'].notna().sum():,}  ({final['Lat'].notna().sum()/len(final)*100:.1f}%)")
log(f"  ❌ No result:     {(final['Display_Name']=='NO_RESULT').sum():,}")
log(f"  ⚠️  Errors:       {final['Display_Name'].str.startswith('ERROR').sum():,}")
log(f"\nResults saved to: {RESULTS_FILE}")
save_log()