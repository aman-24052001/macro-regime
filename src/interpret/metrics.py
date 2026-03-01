from __future__ import annotations
"""
interpret/metrics.py  —  N in OSEMN
--------------------------------------
Full performance tear sheet metrics for the strategy and all benchmarks.

Metrics computed per the project spec:
  Return   : CAGR, total return, rolling 12M
  Risk     : Vol, Max DD, DD duration, VaR-95, CVaR-95, downside dev, Ulcer
  Adjusted : Sharpe, Sortino, Calmar, Omega, IR, Alpha, Beta
  Cost     : turnover, annual TC drag

All annualisation uses 12 (monthly frequency).
Risk-free rate defaults to 0 if not supplied.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

MONTHS_PER_YEAR = 12


class PerformanceMetrics:
    """Compute and format the full performance tear sheet."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    # ── public ───────────────────────────────────────────────────────────

    def compute(
        self,
        returns: pd.Series,
        risk_free_monthly: float = 0.0,
        label: str = "Strategy",
    ) -> dict:
        """
        Compute all metrics for a return series.

        Args:
            returns           : monthly net returns (fraction)
            risk_free_monthly : monthly risk-free rate (e.g. 0.0035 for 4.2% annual)
            label             : name for reporting

        Returns:
            dict of metric_name → value (numeric, not formatted)
        """
        r = returns.dropna()
        n = len(r)
        if n < 2:
            return {"label": label, "error": "insufficient data"}

        # ── Return ────────────────────────────────────────────────────────
        cagr = (1 + r).prod() ** (MONTHS_PER_YEAR / n) - 1
        total_return = (1 + r).prod() - 1

        # ── Risk ──────────────────────────────────────────────────────────
        vol = r.std() * np.sqrt(MONTHS_PER_YEAR)

        cum = (1 + r).cumprod()
        rolling_max = cum.cummax()
        dd = (cum - rolling_max) / rolling_max
        max_dd = float(dd.min())
        dd_dur = _max_dd_duration(dd)

        downside = r[r < risk_free_monthly]
        downside_std = downside.std() * np.sqrt(MONTHS_PER_YEAR) if len(downside) > 1 else vol

        var_95 = float(np.percentile(r, 5))
        cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).any() else var_95

        ulcer = float(np.sqrt((dd ** 2).mean()))

        # ── Risk-adjusted ─────────────────────────────────────────────────
        excess = r - risk_free_monthly
        sharpe = (excess.mean() / excess.std() * np.sqrt(MONTHS_PER_YEAR)
                  if excess.std() > 1e-9 else 0.0)
        sortino = (excess.mean() * MONTHS_PER_YEAR / downside_std
                   if downside_std > 1e-9 else 0.0)
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else np.nan

        # Omega ratio
        threshold = risk_free_monthly
        gains = (r[r > threshold] - threshold).sum()
        losses = (threshold - r[r <= threshold]).sum()
        omega = gains / losses if losses > 1e-9 else np.inf

        # ── Distribution ─────────────────────────────────────────────────
        skewness = float(r.skew())
        excess_kurt = float(r.kurt())
        hit_rate = float((r > 0).mean())

        # ── Calendar ──────────────────────────────────────────────────────
        best_month = float(r.max())
        worst_month = float(r.min())

        return {
            "label": label,
            # Return
            "cagr": cagr,
            "total_return": total_return,
            # Risk
            "volatility": vol,
            "max_drawdown": max_dd,
            "max_dd_duration_months": dd_dur,
            "downside_deviation": downside_std,
            "var_95_monthly": var_95,
            "cvar_95_monthly": cvar_95,
            "ulcer_index": ulcer,
            # Risk-adjusted
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "omega_ratio": omega,
            # Distribution
            "skewness": skewness,
            "excess_kurtosis": excess_kurt,
            "hit_rate": hit_rate,
            "best_month": best_month,
            "worst_month": worst_month,
            # Meta
            "n_months": n,
        }

    def comparison_table(
        self,
        results: dict[str, pd.DataFrame],
        risk_free_monthly: float = 0.0,
    ) -> pd.DataFrame:
        """
        Build a formatted comparison table for all strategies/benchmarks.

        Returns two DataFrames:
            numeric_df  — raw floats for charting
            display_df  — formatted strings for printing / HTML
        """
        rows_numeric: list[dict] = []
        for name, df in results.items():
            m = self.compute(df["return"], risk_free_monthly, label=name)
            rows_numeric.append(m)

        numeric_df = pd.DataFrame(rows_numeric).set_index("label")

        # Pretty-print version
        pct_cols = [
            "cagr", "total_return", "volatility", "max_drawdown",
            "downside_deviation", "var_95_monthly", "cvar_95_monthly",
            "hit_rate", "best_month", "worst_month",
        ]
        display_df = numeric_df.copy().astype(object)
        for col in pct_cols:
            if col in display_df.columns:
                display_df[col] = numeric_df[col].map(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "—"
                )
        for col in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "ulcer_index"]:
            if col in display_df.columns:
                display_df[col] = numeric_df[col].map(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                )

        return numeric_df, display_df

    def rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 36,
        risk_free_monthly: float = 0.0,
    ) -> pd.Series:
        """Rolling Sharpe ratio over a configurable window."""
        excess = returns - risk_free_monthly
        rs = (
            excess.rolling(window).mean()
            / excess.rolling(window).std()
            * np.sqrt(MONTHS_PER_YEAR)
        )
        return rs.rename(f"rolling_{window}m_sharpe")


# ── helpers ──────────────────────────────────────────────────────────────

def _max_dd_duration(drawdown: pd.Series) -> int:
    """Return the maximum consecutive months spent in drawdown."""
    in_dd = drawdown < 0
    max_dur, cur_dur = 0, 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dur
