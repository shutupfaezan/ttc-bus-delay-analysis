"""
investigate_diversion.py — inspect Diversion rows before modelling
Run with: python investigate_diversion.py
"""
import pandas as pd
import numpy as np

INPUT_FILE = "../data/processed/master_ttc_model_ready.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
div = df[df['Incident_Category'] == 'Diversion'].copy()

print(f"Total Diversion rows: {len(div):,}")
print(f"As % of dataset:      {len(div)/len(df)*100:.1f}%\n")

# ── Delay distribution ────────────────────────────────────────────────────────
print("── Delay distribution (Diversion only) ──")
print(f"  Min:    {div['Min_Delay'].min():.0f} min")
print(f"  Median: {div['Min_Delay'].median():.0f} min")
print(f"  Mean:   {div['Min_Delay'].mean():.1f} min")
print(f"  Std:    {div['Min_Delay'].std():.1f} min")
print(f"  Max:    {div['Min_Delay'].max():.0f} min")

print(f"\n  Bucket breakdown:")
buckets = [0, 5, 10, 15, 30, 60, 120, 300]
for i in range(len(buckets) - 1):
    lo, hi = buckets[i], buckets[i+1]
    n   = ((div['Min_Delay'] > lo) & (div['Min_Delay'] <= hi)).sum()
    pct = n / len(div) * 100
    print(f"    {lo:>3}–{hi:<3} min:  {n:>7,}  ({pct:.1f}%)")

# ── Compare to all other categories ──────────────────────────────────────────
print(f"\n── Median delay by category (for context) ──")
print(df.groupby('Incident_Category')['Min_Delay']
      .agg(['median','mean','max'])
      .round(1)
      .sort_values('mean', ascending=False)
      .to_string())

# ── Year breakdown — is it concentrated in certain years? ────────────────────
print(f"\n── Diversion rows by year ──")
print(div.groupby('Year')['Min_Delay']
      .agg(count='count', median='median', mean='mean', max='max')
      .round(1)
      .to_string())

# ── Are extreme values (>60 min) concentrated in specific routes? ─────────────
print(f"\n── Diversion rows with delay > 60 min ──")
extreme = div[div['Min_Delay'] > 60]
print(f"  Count: {len(extreme):,}  ({len(extreme)/len(div)*100:.1f}% of all Diversion rows)")
print(f"\n  Top 10 routes with extreme Diversion delays:")
print(extreme.groupby('Route_Number')['Min_Delay']
      .agg(count='count', median='median', mean='mean', max='max')
      .round(1)
      .sort_values('count', ascending=False)
      .head(10)
      .to_string())

# ── Sample of extreme rows ────────────────────────────────────────────────────
print(f"\n── Sample of 20 Diversion rows with delay > 60 min ──")
sample_cols = ['Service_Date', 'Route_Number', 'Hour', 'Min_Delay',
               'Temp_C', 'Visibility_km', 'Year']
sample_cols = [c for c in sample_cols if c in extreme.columns]
print(extreme[sample_cols]
      .sort_values('Min_Delay', ascending=False)
      .head(20)
      .to_string(index=False))

# ── What % of total delay minutes do extreme Diversion rows account for? ──────
total_delay  = df['Min_Delay'].sum()
extreme_delay = extreme['Min_Delay'].sum()
print(f"\n── Impact of extreme Diversion rows ──")
print(f"  All rows total delay:           {total_delay:,.0f} min")
print(f"  Diversion >60 min total delay:  {extreme_delay:,.0f} min  ({extreme_delay/total_delay*100:.1f}% of all delay minutes)")
print(f"  Diversion >60 min row count:    {len(extreme):,}  ({len(extreme)/len(df)*100:.2f}% of all rows)")