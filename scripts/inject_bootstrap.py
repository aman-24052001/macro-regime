#!/usr/bin/env python3
"""
scripts/inject_bootstrap.py
-----------------------------
Reads outputs/reports/bootstrap_results.csv and injects the data
into outputs/dashboard.html as a JS object (const bootData = {...}).

Run after `python main.py --phase bootstrap` completes.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CSV  = ROOT / "outputs" / "reports" / "bootstrap_results.csv"
HTML = ROOT / "outputs" / "dashboard.html"

if not CSV.exists():
    print(f"ERROR: {CSV} not found. Run --phase bootstrap first.")
    sys.exit(1)

import pandas as pd
df = pd.read_csv(CSV)
row = df.iloc[0].to_dict()

# Build JS-safe record
record = {k: (float(v) if isinstance(v, (float, int)) else v) for k, v in row.items()}
js_data = "const bootData = " + json.dumps(record, indent=2) + ";"

html = HTML.read_text(encoding="utf-8")

OLD = "const bootData = null; // PLACEHOLDER — replaced when bootstrap results arrive"
if OLD not in html:
    print("ERROR: placeholder not found in dashboard.html. Already injected?")
    sys.exit(1)

html = html.replace(OLD, js_data)
HTML.write_text(html, encoding="utf-8")

print(f"Injected bootstrap results into dashboard.html")
print(f"\nSharpe : {row['sharpe_point']:.3f}  95% CI [{row['sharpe_ci_lo']:.3f}, {row['sharpe_ci_hi']:.3f}]")
print(f"CAGR   : {row['cagr_point']:.1%}  95% CI [{row['cagr_ci_lo']:.1%}, {row['cagr_ci_hi']:.1%}]")
print(f"Max DD : {row['maxdd_point']:.1%}  95% CI [{row['maxdd_ci_lo']:.1%}, {row['maxdd_ci_hi']:.1%}]")
print(f"p(Sharpe > SPY {row['spy_sharpe']:.3f}) = {row['sharpe_pval_vs_spy']:.3f}")
