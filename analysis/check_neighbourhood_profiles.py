"""
check_neighbourhood_profiles.py — inspect neighbourhood profile structure
Run with: python check_neighbourhood_profiles.py  (from analysis/ folder)
"""
import pandas as pd

df = pd.read_excel(
    "../data/raw/Neighbour Profiles/neighbourhood-profiles-2021-158-model (1).xlsx"
)

print(f"Shape: {df.shape}")
print(f"First column name: '{df.columns[0]}'")
print(f"\nFirst 5 rows of first column (row labels):")
print(df.iloc[:5, 0].to_string())

# Search for population and density related rows
print(f"\n── Rows containing 'population' (case insensitive) ──")
pop_mask = df.iloc[:, 0].astype(str).str.lower().str.contains('population')
pop_rows = df[pop_mask]
print(f"Found {len(pop_rows)} rows")
for idx, row in pop_rows.iterrows():
    print(f"  Row {idx:>4}: {row.iloc[0]}")

print(f"\n── Rows containing 'density' (case insensitive) ──")
den_mask = df.iloc[:, 0].astype(str).str.lower().str.contains('density')
den_rows = df[den_mask]
print(f"Found {len(den_rows)} rows")
for idx, row in den_rows.iterrows():
    print(f"  Row {idx:>4}: {row.iloc[0]}")

print(f"\n── Rows containing 'area' (case insensitive) ──")
area_mask = df.iloc[:, 0].astype(str).str.lower().str.contains('area')
area_rows = df[area_mask]
print(f"Found {len(area_rows)} rows")
for idx, row in area_rows.iterrows():
    print(f"  Row {idx:>4}: {row.iloc[0]}")

print(f"\n── Sample: population row values (first 5 neighbourhoods) ──")
if len(pop_rows) > 0:
    first_pop = pop_rows.iloc[0]
    print(f"Row label: {first_pop.iloc[0]}")
    print(f"First 5 neighbourhood values:")
    print(first_pop.iloc[1:6].to_string())
