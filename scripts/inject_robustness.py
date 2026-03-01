#!/usr/bin/env python3
"""
scripts/inject_robustness.py
-----------------------------
Reads outputs/reports/robustness_grid.csv and injects the data
into outputs/dashboard.html as a JS array (const robData = [...]).

Run after `python main.py --phase robustness` completes.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CSV  = ROOT / "outputs" / "reports" / "robustness_grid.csv"
HTML = ROOT / "outputs" / "dashboard.html"

if not CSV.exists():
    print(f"ERROR: {CSV} not found. Run --phase robustness first.")
    sys.exit(1)

import pandas as pd
df = pd.read_csv(CSV)

# Build JS-safe record list
records = []
for _, row in df.iterrows():
    records.append({
        "K":             int(row["K"]),
        "train_years":   int(row["train_years"]),
        "tc_bps":        float(row["tc_bps"]),
        "threshold_pct": float(row["threshold_pct"]),
        "sharpe":        round(float(row["sharpe"]),  4),
        "sortino":       round(float(row["sortino"]), 4),
        "calmar":        round(float(row["calmar"]),  4),
        "cagr":          round(float(row["cagr"]),    4),
        "max_dd":        round(float(row["max_dd"]),  4),
        "volatility":    round(float(row["volatility"]), 4),
    })

js_data = "const robData = " + json.dumps(records, indent=2) + ";"

html = HTML.read_text(encoding="utf-8")

# Replace the placeholder line
OLD = "const robData = []; // PLACEHOLDER — replaced when grid results arrive"
if OLD not in html:
    print("ERROR: placeholder not found in dashboard.html. Already injected?")
    sys.exit(1)

html = html.replace(OLD, js_data)
HTML.write_text(html, encoding="utf-8")

print(f"Injected {len(records)} grid rows into dashboard.html")

# Print summary stats
print(f"\nSharpe range : {df['sharpe'].min():.3f} – {df['sharpe'].max():.3f}")
print(f"Beats SPY    : {(df['sharpe'] >= 0.760).sum()}/{len(df)} "
      f"({(df['sharpe'] >= 0.760).mean()*100:.0f}%)")
print(f"Worst case   : K={int(df.loc[df['sharpe'].idxmin(),'K'])} "
      f"train={int(df.loc[df['sharpe'].idxmin(),'train_years'])}yr "
      f"TC={df.loc[df['sharpe'].idxmin(),'tc_bps']}bps "
      f"thr={df.loc[df['sharpe'].idxmin(),'threshold_pct']}%  "
      f"Sharpe={df['sharpe'].min():.3f}")
print(f"Best  case   : K={int(df.loc[df['sharpe'].idxmax(),'K'])} "
      f"train={int(df.loc[df['sharpe'].idxmax(),'train_years'])}yr "
      f"TC={df.loc[df['sharpe'].idxmax(),'tc_bps']}bps "
      f"thr={df.loc[df['sharpe'].idxmax(),'threshold_pct']}%  "
      f"Sharpe={df['sharpe'].max():.3f}")
