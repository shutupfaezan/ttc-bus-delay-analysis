"""
prepare_eda_master.py — TTC Master EDA Dataset Builder
Joins all data sources into one clean CSV for team EDA work.
Columns: delay records + weather + incident category + headway +
         geocoded locations + neighbourhood + population density
Run with: python prepare_eda_master.py  (from pipeline/ folder)
Wait for geocode_nominatim_filtered.py to complete before running.
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
EDA_FILE        = "../data/processed/master_ttc_eda_ready.csv"
HEADWAY_FILE    = "../data/processed/gtfs_headway_lookup.csv"
GEOCODED_FILE   = "../data/geocoding/geocoded_locations.csv"
GEOJSON_FILE    = "../data/raw/Neighbour Geojson/Neighbourhoods - 4326 (1).geojson"
PROFILE_FILE    = "../data/raw/Neighbour Profiles/neighbourhood-profiles-2021-158-model (1).xlsx"
OUTPUT_FILE     = "../data/processed/master_ttc_eda_master.csv"
LOG_FILE        = "../logs/prepare_eda_master_log.txt"
# ─────────────────────────────────────────────────────────────────────────────

log_lines = []
def log(text=""):
    print(text)
    log_lines.append(str(text))

log(f"prepare_eda_master started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── INCIDENT CODE MAPPING ─────────────────────────────────────────────────────
INCIDENT_MAP = {
    'Mechanical': 'Mechanical', 'MFO': 'Mechanical', 'MFSH': 'Mechanical',
    'MFS': 'Mechanical', 'MFFD': 'Mechanical', 'MFPR': 'Mechanical',
    'MFWEA': 'Mechanical', 'MFLD': 'Mechanical',
    'Late Entering Service - Mechanical': 'Mechanical',
    'MTO': 'Mechanical', 'MTUS': 'Mechanical', 'MTIE': 'Mechanical',
    'Diversion': 'Diversion', 'MFDV': 'Diversion',
    'EFD': 'Diversion', 'MTDV': 'Diversion',
    'Utilized Off Route': 'Utilized Off Route',
    'Utilizing Off Route': 'Utilized Off Route',
    'MFTO': 'Utilized Off Route', 'MFUI': 'Utilized Off Route',
    'MFUIR': 'Utilized Off Route', 'MUO': 'Utilized Off Route',
    'MUIE': 'Utilized Off Route', 'MUIS': 'Utilized Off Route',
    'TFO': 'Traffic', 'TFCNO': 'Traffic', 'TFPD': 'Traffic',
    'TFOI': 'Traffic', 'TFLF': 'Traffic', 'TFLL': 'Traffic',
    'TFPI': 'Traffic',
    'General Delay': 'General Delay', 'Held By': 'General Delay',
    'Late': 'General Delay', 'Operations': 'General Delay',
    'SUDP': 'General Delay',
    'Late Leaving Garage': 'Late Departure',
    'Late Leaving Garage - Mechanical': 'Late Departure',
    'Late Leaving Garage - Operator': 'Late Departure',
    'Late Leaving Garage - Management': 'Late Departure',
    'Late Leaving Garage - Vision': 'Late Departure',
    'Late Leaving Garage - Operations': 'Late Departure',
    'Late Entering Service': 'Late Departure',
    'PREL': 'Late Departure', 'MFESA': 'Late Departure',
    'Investigation': 'Investigation',
    'Emergency Services': 'Emergency Services',
    'EFO': 'Emergency Services', 'EFB': 'Emergency Services',
    'EFHVA': 'Emergency Services', 'EFRA': 'Emergency Services',
    'EFCAN': 'Emergency Services', 'ETO': 'Emergency Services',
    'Security': 'Security', 'SFDP': 'Security', 'SFO': 'Security',
    'SFPOL': 'Security', 'SFAP': 'Security', 'SFAE': 'Security',
    'SFSA': 'Security', 'SFSP': 'Security', 'STO': 'Security',
    'SRO': 'Security', 'Securitty': 'Security',
    'Collision - TTC': 'Collision',
    'Collision - TTC Involved': 'Collision',
    'Road Blocked - NON-TTC Collision': 'Collision',
    'Road Block - Non-TTC Collision': 'Collision',
    'Roadblock by Collision - Non-TTC': 'Collision',
    'TTPD': 'Collision',
    'Operations - Operator': 'Operator', 'Management': 'Operator',
    'Cleaning': 'Cleaning', 'Cleaning - Unsanitary': 'Cleaning',
    'Cleaning - Disinfection': 'Cleaning',
    'MFSAN': 'Cleaning', 'MFUS': 'Cleaning', 'MTSAN': 'Cleaning',
    'Vision': 'Vision', 'MFVIS': 'Vision', 'MTVIS': 'Vision',
    'PFO': 'Passenger', 'PFPD': 'Passenger',
    'EFP': 'Passenger', 'MFPI': 'Passenger',
    'Overhead': 'Other', 'Rail/Switches': 'Other',
    'e': 'Other', 'MTNOA': 'Other',
}

# ── STEP 1 — LOAD BASE DATA ───────────────────────────────────────────────────
log("\n[ STEP 1 — LOAD BASE DATA ]")
df = pd.read_csv(EDA_FILE, low_memory=False)
log(f"  Rows loaded: {len(df):,}  |  Columns: {len(df.columns)}")

# ── STEP 2 — SELECT AND CLEAN COLUMNS ────────────────────────────────────────
log("\n[ STEP 2 — SELECT COLUMNS ]")
KEEP = [
    'Service_Date', 'Route_Number', 'Day', 'Hour', 'Month', 'Year',
    'Season', 'Is_Rush_Hour', 'Is_Weekend', 'Min_Delay', 'Is_Severe',
    'Incident_Code', 'Vehicle', 'Location',
    'Temp_C', 'Visibility_km', 'Wind_Spd_kmh', 'Rel_Humidity_pct',
]
df = df[[c for c in KEEP if c in df.columns]].copy()
log(f"  Columns selected: {list(df.columns)}")

# ── STEP 3 — DERIVE COLUMNS ───────────────────────────────────────────────────
log("\n[ STEP 3 — DERIVE COLUMNS ]")

# Incident category
df['Incident_Category'] = df['Incident_Code'].map(INCIDENT_MAP)
mapped   = df['Incident_Category'].notna().sum()
unmapped = df['Incident_Category'].isna().sum()
log(f"  Incident_Category: {mapped:,} mapped  |  {unmapped:,} unmapped (kept as null)")

# Severe flags
df['Is_Severe']    = (df['Min_Delay'] >= 15).astype(int)
df['Is_Severe_30'] = (df['Min_Delay'] >= 30).astype(int)
log(f"  Is_Severe (>=15 min):  {df['Is_Severe'].sum():,} rows ({df['Is_Severe'].mean()*100:.1f}%)")
log(f"  Is_Severe_30 (>=30 min): {df['Is_Severe_30'].sum():,} rows ({df['Is_Severe_30'].mean()*100:.1f}%)")

# Boolean encoding
df['Is_Rush_Hour'] = df['Is_Rush_Hour'].astype(int)
df['Is_Weekend']   = df['Is_Weekend'].astype(int)

# Clean Vehicle — null out 0.0 placeholder
df['Vehicle'] = df['Vehicle'].replace(0.0, np.nan)
df['Vehicle'] = df['Vehicle'].replace('0.0', np.nan)
log(f"  Vehicle: {df['Vehicle'].notna().sum():,} non-null  |  {df['Vehicle'].isna().sum():,} null")

# ── STEP 4 — JOIN HEADWAY ─────────────────────────────────────────────────────
log("\n[ STEP 4 — JOIN HEADWAY ]")
headway = pd.read_csv(HEADWAY_FILE)
headway['Route_Number'] = headway['Route_Number'].astype(float)
df['Route_Number']      = pd.to_numeric(df['Route_Number'], errors='coerce')
before = len(df)
df = df.merge(headway, on=['Route_Number', 'Hour'], how='left')
global_median_hw        = headway['Headway_min'].median()
null_hw                 = df['Headway_min'].isna().sum()
df['Headway_min']       = df['Headway_min'].fillna(global_median_hw)
log(f"  Rows before/after: {before:,} / {len(df):,}")
log(f"  Null headway filled with median ({global_median_hw:.1f} min): {null_hw:,} rows")

# ── STEP 5 — JOIN GEOCODED LOCATIONS ─────────────────────────────────────────
log("\n[ STEP 5 — JOIN GEOCODED LOCATIONS ]")
if not os.path.exists(GEOCODED_FILE):
    log(f"  WARNING: {GEOCODED_FILE} not found — skipping geocoding join")
    log(f"  Run geocode_nominatim_filtered.py first then rerun this script")
    df['Lat'] = np.nan
    df['Lon'] = np.nan
else:
    geo = pd.read_csv(GEOCODED_FILE)
    geo = geo.dropna(subset=['Lat', 'Lon'])
    geo = geo[['Location_Norm', 'Lat', 'Lon']].drop_duplicates('Location_Norm')
    log(f"  Geocoded locations loaded: {len(geo):,} with valid coords")

    # Normalise location for join
    df['Location_Norm'] = df['Location'].astype(str).str.strip().str.upper()
    geo['Location_Norm'] = geo['Location_Norm'].astype(str).str.strip().str.upper()

    df = df.merge(geo, on='Location_Norm', how='left')
    df = df.drop(columns=['Location_Norm'])

    matched = df['Lat'].notna().sum()
    log(f"  Rows with geocoords: {matched:,} ({matched/len(df)*100:.1f}%)")

# ── STEP 6 — SPATIAL JOIN TO NEIGHBOURHOODS ───────────────────────────────────
log("\n[ STEP 6 — SPATIAL JOIN TO NEIGHBOURHOODS ]")
if df['Lat'].notna().sum() == 0:
    log("  Skipping — no geocoords available")
    df['Neighbourhood'] = np.nan
else:
    try:
        from shapely.geometry import Point, shape

        with open(GEOJSON_FILE, encoding='utf-8') as f:
            gj = json.load(f)

        # Build list of (neighbourhood_name, shapely_shape)
        neighbourhoods = []
        for feature in gj['features']:
            name  = feature['properties']['AREA_NAME']
            geom  = shape(feature['geometry'])
            neighbourhoods.append((name, geom))

        log(f"  Neighbourhood polygons loaded: {len(neighbourhoods)}")

        def find_neighbourhood(lat, lon):
            if pd.isna(lat) or pd.isna(lon):
                return np.nan
            pt = Point(lon, lat)
            for name, geom in neighbourhoods:
                if geom.contains(pt):
                    return name
            return np.nan

        log("  Assigning neighbourhoods (this may take 1-2 min) ...")
        geo_rows = df['Lat'].notna()
        df.loc[geo_rows, 'Neighbourhood'] = df.loc[geo_rows].apply(
            lambda r: find_neighbourhood(r['Lat'], r['Lon']), axis=1
        )
        df.loc[~geo_rows, 'Neighbourhood'] = np.nan

        assigned = df['Neighbourhood'].notna().sum()
        log(f"  Rows assigned to neighbourhood: {assigned:,} ({assigned/len(df)*100:.1f}%)")

    except ImportError:
        log("  WARNING: shapely not installed — run: pip install shapely")
        log("  Skipping neighbourhood spatial join")
        df['Neighbourhood'] = np.nan

# ── STEP 7 — JOIN POPULATION DATA ────────────────────────────────────────────
log("\n[ STEP 7 — JOIN POPULATION DATA ]")
try:
    profile = pd.read_excel(PROFILE_FILE)

    # Row 2 = total population (0-indexed row 2 in the dataframe)
    pop_row = profile[profile.iloc[:, 0].astype(str).str.contains(
        'Total - Age groups of the population', na=False
    )].iloc[0]

    # Transpose: neighbourhood names become index, value is population
    neighbourhood_names = profile.columns[1:].tolist()
    populations         = pop_row.iloc[1:].tolist()

    pop_df = pd.DataFrame({
        'Neighbourhood': neighbourhood_names,
        'Population':    pd.to_numeric(populations, errors='coerce')
    })
    log(f"  Population data loaded: {len(pop_df):,} neighbourhoods")
    log(f"  Population range: {pop_df['Population'].min():.0f} → {pop_df['Population'].max():.0f}")

    # ── Compute area from GeoJSON polygons ────────────────────────────────────
    try:
        import pyproj
        from shapely.geometry import shape
        from shapely.ops import transform
        from functools import partial

        with open(GEOJSON_FILE, encoding='utf-8') as f:
            gj = json.load(f)

        # Project to UTM zone 17N (Toronto) for accurate area in m²
        wgs84  = pyproj.CRS('EPSG:4326')
        utm17n = pyproj.CRS('EPSG:32617')
        project = pyproj.Transformer.from_crs(wgs84, utm17n,
                                               always_xy=True).transform

        area_records = []
        for feature in gj['features']:
            name = feature['properties']['AREA_NAME']
            geom = shape(feature['geometry'])
            geom_proj  = transform(project, geom)
            area_km2   = geom_proj.area / 1_000_000
            area_records.append({'Neighbourhood': name, 'Area_km2': round(area_km2, 4)})

        area_df = pd.DataFrame(area_records)
        log(f"  Area computed for {len(area_df)} neighbourhoods")
        log(f"  Area range: {area_df['Area_km2'].min():.2f} → {area_df['Area_km2'].max():.2f} km²")

        # Join population + area
        density_df = pop_df.merge(area_df, on='Neighbourhood', how='inner')
        density_df['Population_Density'] = (
            density_df['Population'] / density_df['Area_km2']
        ).round(1)
        log(f"  Density range: {density_df['Population_Density'].min():.0f} → "
            f"{density_df['Population_Density'].max():.0f} people/km²")

        # Join to main dataframe
        if 'Neighbourhood' in df.columns and df['Neighbourhood'].notna().sum() > 0:
            df = df.merge(
                density_df[['Neighbourhood', 'Population', 'Area_km2', 'Population_Density']],
                on='Neighbourhood', how='left'
            )
            joined = df['Population_Density'].notna().sum()
            log(f"  Rows with population density: {joined:,} ({joined/len(df)*100:.1f}%)")
        else:
            log("  Neighbourhood column empty — skipping density join")
            df['Population']        = np.nan
            df['Area_km2']          = np.nan
            df['Population_Density'] = np.nan

    except ImportError:
        log("  WARNING: pyproj not installed — run: pip install pyproj")
        log("  Skipping area/density calculation")
        df['Population']         = np.nan
        df['Area_km2']           = np.nan
        df['Population_Density'] = np.nan

except Exception as e:
    log(f"  ERROR loading population data: {e}")
    df['Population']         = np.nan
    df['Area_km2']           = np.nan
    df['Population_Density'] = np.nan

# ── STEP 8 — FINAL COLUMN ORDER ───────────────────────────────────────────────
log("\n[ STEP 8 — FINAL COLUMN ORDER ]")
FINAL_COLS = [
    # Identity
    'Service_Date', 'Year', 'Month', 'Day', 'Hour', 'Season',
    # Route
    'Route_Number', 'Headway_min',
    # Incident
    'Incident_Code', 'Incident_Category', 'Vehicle',
    # Delay target
    'Min_Delay', 'Is_Severe', 'Is_Severe_30',
    # Time flags
    'Is_Rush_Hour', 'Is_Weekend',
    # Weather
    'Temp_C', 'Visibility_km', 'Wind_Spd_kmh', 'Rel_Humidity_pct',
    # Location
    'Location', 'Lat', 'Lon',
    # Neighbourhood
    'Neighbourhood', 'Population', 'Area_km2', 'Population_Density',
]
df = df[[c for c in FINAL_COLS if c in df.columns]]
log(f"  Final columns ({len(df.columns)}): {list(df.columns)}")

# ── STEP 9 — NULL SUMMARY ─────────────────────────────────────────────────────
log("\n[ STEP 9 — NULL SUMMARY ]")
for col in df.columns:
    n   = df[col].isna().sum()
    pct = n / len(df) * 100
    if n > 0:
        log(f"  {col:<25}  nulls: {n:>8,}  ({pct:.1f}%)")

# ── SAVE ──────────────────────────────────────────────────────────────────────
log("\n[ SAVING ]")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
sz = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
log(f"  Saved: {OUTPUT_FILE}  ({sz:.1f} MB)")
log(f"  Rows:    {len(df):,}")
log(f"  Columns: {len(df.columns)}")

log(f"\n{'='*60}")
log("MASTER EDA CSV READY")
log(f"{'='*60}")
log(f"  File: master_ttc_eda_master.csv")
log(f"  This is the single file for all team EDA work.")
log(f"  Covers: PBI 1, 2, 3, Vehicle Intel, YoY Trend,")
log(f"          Incident Treemap, Severe Rate, Top N Routes")
log(f"  Neighbourhood columns populated only after geocoding completes.")

log(f"\nprepare_eda_master complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Log saved: {LOG_FILE}")