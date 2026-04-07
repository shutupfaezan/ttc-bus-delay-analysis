"""
join_gtfs.py — TTC GTFS Headway Extraction (buses only)
Filters to route_type == 3 (bus) before computing headway.
Excludes subways (route_type 1) and streetcars (route_type 0).
Computes average headway (minutes between buses) per route per hour of day.
Run with: python join_gtfs.py  (from pipeline/ folder)
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
GTFS_DIR      = "../data/raw/GTFS"
INPUT_FILE    = "../data/processed/master_ttc_model_ready.csv"
OUTPUT_FILE   = "../data/processed/master_ttc_model_ready.csv"
HEADWAY_FILE  = "../data/processed/gtfs_headway_lookup.csv"
LOG_FILE      = "../logs/join_gtfs_log.txt"
# ─────────────────────────────────────────────────────────────────────────────

log_lines = []
def log(text=""):
    print(text)
    log_lines.append(str(text))

log(f"join_gtfs started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── LOAD GTFS FILES ───────────────────────────────────────────────────────────
log("\n[ LOADING GTFS FILES ]")

# routes.txt — filter to buses only (route_type == 3)
routes = pd.read_csv(
    os.path.join(GTFS_DIR, "routes.txt"),
    usecols=["route_id", "route_short_name", "route_type"],
    dtype=str
)
log(f"  routes.txt total:    {len(routes):,} routes")

# Show breakdown by route type before filtering
routes["route_type_int"] = pd.to_numeric(routes["route_type"], errors="coerce")
type_map = {0: "streetcar", 1: "subway", 3: "bus"}
for rtype, label in sorted(type_map.items()):
    n = (routes["route_type_int"] == rtype).sum()
    log(f"    route_type {rtype} ({label}): {n}")

# Keep buses only
routes = routes[routes["route_type_int"] == 3].copy()
routes["route_short_name"] = pd.to_numeric(routes["route_short_name"], errors="coerce")
routes = routes.dropna(subset=["route_short_name"])
routes["route_short_name"] = routes["route_short_name"].astype(int)
log(f"  Bus routes kept:     {len(routes):,}")
log(f"  Bus route range:     {routes['route_short_name'].min()} → {routes['route_short_name'].max()}")

# trips.txt
trips = pd.read_csv(
    os.path.join(GTFS_DIR, "trips.txt"),
    usecols=["route_id", "trip_id"],
    dtype=str
)
log(f"  trips.txt:           {len(trips):,} trips loaded")

# Filter trips to bus routes only before loading stop_times
bus_route_ids = set(routes["route_id"].astype(str))
trips = trips[trips["route_id"].isin(bus_route_ids)].copy()
log(f"  Bus trips kept:      {len(trips):,}")

# stop_times.txt — first stop per trip only, chunked
log(f"  stop_times.txt:      loading in chunks (204MB) ...")
bus_trip_ids = set(trips["trip_id"].astype(str))
chunks = []
for chunk in pd.read_csv(
    os.path.join(GTFS_DIR, "stop_times.txt"),
    usecols=["trip_id", "arrival_time", "stop_sequence"],
    dtype={"trip_id": str, "arrival_time": str, "stop_sequence": "Int64"},
    chunksize=500_000,
):
    first = chunk[
        (chunk["trip_id"].isin(bus_trip_ids)) &
        (chunk["stop_sequence"] == 1)
    ][["trip_id", "arrival_time"]]
    if len(first) > 0:
        chunks.append(first)

stop_times = pd.concat(chunks, ignore_index=True)
log(f"  Bus first-stop departures: {len(stop_times):,}")

# ── PARSE DEPARTURE HOURS ─────────────────────────────────────────────────────
log("\n[ PARSING DEPARTURE HOURS ]")

def parse_gtfs_hour(time_str):
    try:
        hour = int(str(time_str).split(":")[0])
        return hour % 24
    except:
        return np.nan

stop_times["Hour"] = stop_times["arrival_time"].apply(parse_gtfs_hour)
stop_times = stop_times.dropna(subset=["Hour"])
stop_times["Hour"] = stop_times["Hour"].astype(int)
log(f"  Unique hours parsed: {stop_times['Hour'].nunique()}")

# ── JOIN: stop_times → trips → routes ────────────────────────────────────────
log("\n[ JOINING GTFS TABLES ]")

st_with_route = stop_times.merge(trips, on="trip_id", how="left")
st_with_num   = st_with_route.merge(
    routes[["route_id", "route_short_name"]], on="route_id", how="left"
)
st_with_num = st_with_num.dropna(subset=["route_short_name"])
st_with_num = st_with_num.rename(columns={"route_short_name": "Route_Number"})

log(f"  Final joined rows:   {len(st_with_num):,}")
log(f"  Unique bus routes:   {st_with_num['Route_Number'].nunique()}")

log(f"\n  Hour distribution (bus departures only):")
for hr, cnt in st_with_num.groupby("Hour").size().items():
    log(f"    Hour {hr:02d}: {cnt:,} scheduled bus departures")

# ── COMPUTE HEADWAY PER ROUTE PER HOUR ───────────────────────────────────────
log("\n[ COMPUTING HEADWAY ]")

def compute_headway(group):
    times = group["arrival_time"].dropna().unique()
    mins  = []
    for t in times:
        try:
            parts = str(t).split(":")
            m = int(parts[0]) * 60 + int(parts[1])
            mins.append(m)
        except:
            continue
    if len(mins) < 2:
        return np.nan
    mins_sorted = sorted(set(mins))
    gaps = [mins_sorted[i+1] - mins_sorted[i] for i in range(len(mins_sorted)-1)]
    gaps = [g for g in gaps if 0 < g <= 120]
    return np.mean(gaps) if gaps else np.nan

log("  Computing per-route per-hour headway ...")
headway_df = (
    st_with_num
    .groupby(["Route_Number", "Hour"])
    .apply(compute_headway, include_groups=False)
    .reset_index()
    .rename(columns={0: "Headway_min"})
)

headway_df = headway_df.dropna(subset=["Headway_min"])
headway_df["Headway_min"] = headway_df["Headway_min"].round(1)

log(f"  Headway lookup rows: {len(headway_df):,}  (bus route × hour combinations)")
log(f"  Headway range:       {headway_df['Headway_min'].min():.1f} → {headway_df['Headway_min'].max():.1f} min")
log(f"  Mean headway:        {headway_df['Headway_min'].mean():.1f} min")
log(f"  Median headway:      {headway_df['Headway_min'].median():.1f} min")

log(f"\n  Top 10 least frequent bus routes (highest headway):")
route_avg = headway_df.groupby("Route_Number")["Headway_min"].mean().sort_values(ascending=False)
for route, hw in route_avg.head(10).items():
    log(f"    Route {route:>5.0f}:  {hw:.1f} min avg headway")

log(f"\n  Top 10 most frequent bus routes (lowest headway):")
for route, hw in route_avg.tail(10).items():
    log(f"    Route {route:>5.0f}:  {hw:.1f} min avg headway")

headway_df.to_csv(HEADWAY_FILE, index=False)
log(f"\n  Headway lookup saved: {HEADWAY_FILE}")

# ── JOIN HEADWAY TO MODEL DATA ────────────────────────────────────────────────
log("\n[ JOINING HEADWAY TO MODEL DATA ]")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Drop stale Headway_min if present from previous run
if "Headway_min" in df.columns:
    df = df.drop(columns=["Headway_min"])
    log(f"  Dropped stale Headway_min column from previous run")

log(f"  Model-ready rows:    {len(df):,}")

df = df.merge(headway_df, on=["Route_Number", "Hour"], how="left")

global_median_headway = headway_df["Headway_min"].median()
null_headway = df["Headway_min"].isna().sum()
df["Headway_min"] = df["Headway_min"].fillna(global_median_headway)
log(f"  Null headway filled with bus median ({global_median_headway:.1f} min): {null_headway:,} rows")
log(f"  Final null headway:  {df['Headway_min'].isna().sum()}")

# ── VERIFY ────────────────────────────────────────────────────────────────────
log("\n[ BUS HEADWAY FEATURE SUMMARY ]")
log(f"  Min:    {df['Headway_min'].min():.1f} min")
log(f"  Median: {df['Headway_min'].median():.1f} min")
log(f"  Mean:   {df['Headway_min'].mean():.1f} min")
log(f"  Max:    {df['Headway_min'].max():.1f} min")

# ── SAVE ─────────────────────────────────────────────────────────────────────
log("\n[ SAVING ]")
df.to_csv(OUTPUT_FILE, index=False)
sz = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
log(f"  Saved: {OUTPUT_FILE}  ({sz:.1f} MB)")
log(f"  Rows:    {len(df):,}")
log(f"  Columns: {len(df.columns)}")
log(f"  Columns: {list(df.columns)}")

log(f"\njoin_gtfs complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Next step: run train_model_v4.py")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"Log saved: {LOG_FILE}")