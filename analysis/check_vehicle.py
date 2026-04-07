"""
check_vehicle.py — inspect Vehicle column for EDA viability
Run with: python check_vehicle.py  (from analysis/ folder)
"""
import pandas as pd

df = pd.read_csv("../data/processed/master_ttc_eda_ready.csv", low_memory=False)

print(f"Total rows: {len(df):,}")
print(f"Vehicle nulls: {df['Vehicle'].isna().sum():,} ({df['Vehicle'].isna().mean()*100:.1f}%)")

print(f"\nVehicle nulls by year:")
print(df.groupby('Year')['Vehicle'].apply(lambda x: x.isna().sum()).to_string())

# Mechanical incident codes
MECHANICAL_CODES = [
    'Mechanical', 'MFO', 'MFSH', 'MFS', 'MFFD', 'MFPR',
    'MFWEA', 'MFLD', 'MTO', 'MTUS', 'MTIE',
    'Late Entering Service - Mechanical'
]

mech = df[df['Incident_Code'].isin(MECHANICAL_CODES)].copy()

print(f"\nMechanical rows:")
print(f"  Total:          {len(mech):,}")
print(f"  With Vehicle ID: {mech['Vehicle'].notna().sum():,} ({mech['Vehicle'].notna().mean()*100:.1f}%)")

print(f"\nTop 20 vehicles by mechanical incidents:")
top = (
    mech[mech['Vehicle'].notna()]
    .groupby('Vehicle')['Min_Delay']
    .agg(count='count', avg_delay='mean', total_delay='sum')
    .round(1)
    .sort_values('count', ascending=False)
    .head(20)
)
print(top.to_string())