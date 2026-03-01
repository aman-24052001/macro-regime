from __future__ import annotations
"""
interpret/backtest.py  —  N in OSEMN
---------------------------------------
Runs the strategy and all benchmarks through the backtest engine,
builds equity curves, and analyses crisis-period performance.

Benchmarks are defined in config — no ticker lists hardcoded here.
Transaction costs are applied consistently across strategy and benchmarks.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Backtester:
    """Compare strategy equity curve against all configured benchmarks."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        bt = cfg["backtest"]

        self.initial_capital: float = bt["initial_capital"]
        self.benchmarks_config: list[dict] = bt["benchmarks"]
        self.crisis_periods: list[dict] = bt["crisis_periods"]

        reb = cfg["portfolio"]["rebalancing"]
        # Benchmark round-trip TC assumption (same as strategy)
        self.bench_tc: float = (
            reb["transaction_cost_bps"] + reb["slippage_bps"]
        ) / 10_000.0

        self.assets: list[str] = list(cfg["data"]["market"]["etfs"].keys())

    # ── public ───────────────────────────────────────────────────────────

    def run(
        self,
        strategy_returns: pd.Series,
        asset_returns: pd.DataFrame,
        inv_vol_weights: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Build equity curves for strategy and all benchmarks.

        Args:
            strategy_returns : net monthly returns from Rebalancer
            asset_returns    : monthly asset returns (date × ticker)
            inv_vol_weights  : optional dynamic inverse-vol benchmark weights

        Returns:
            dict  name → DataFrame(return, cumulative_return, equity_curve)
        """
        results: dict[str, pd.DataFrame] = {}

        # Strategy
        results["Regime Strategy"] = self._equity_curve(strategy_returns.dropna())

        # Static benchmarks from config
        for bench in self.benchmarks_config:
            if bench["weights"] is None:
                # Inverse-vol dynamic benchmark
                if inv_vol_weights is not None:
                    inv_vol_ret = self._dynamic_benchmark_returns(
                        inv_vol_weights, asset_returns
                    )
                    results[bench["name"]] = self._equity_curve(inv_vol_ret)
                continue

            bench_ret = self._static_benchmark_returns(bench["weights"], asset_returns)
            results[bench["name"]] = self._equity_curve(bench_ret)

        return results

    def crisis_analysis(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Return/drawdown for each strategy during each crisis period.
        Shows how the strategy protects capital vs benchmarks.
        """
        rows: list[dict] = []
        for crisis in self.crisis_periods:
            start = pd.Timestamp(crisis["start"])
            end = pd.Timestamp(crisis["end"])

            for name, df in results.items():
                mask = (df.index >= start) & (df.index <= end)
                if mask.sum() < 2:
                    continue
                r = df.loc[mask, "return"]
                cum = df.loc[mask, "cumulative_return"]

                total_ret = (1 + r).prod() - 1
                max_dd = _max_drawdown(cum)
                vol = r.std() * np.sqrt(12) if len(r) > 2 else np.nan

                rows.append(
                    {
                        "crisis": crisis["name"],
                        "strategy": name,
                        "total_return": total_ret,
                        "max_drawdown": max_dd,
                        "annualised_vol": vol,
                        "n_months": mask.sum(),
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["total_return"] = df["total_return"].map("{:.1%}".format)
            df["max_drawdown"] = df["max_drawdown"].map("{:.1%}".format)
            df["annualised_vol"] = df["annualised_vol"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
        return df

    # ── private ──────────────────────────────────────────────────────────

    def _static_benchmark_returns(
        self, weights: dict, asset_returns: pd.DataFrame
    ) -> pd.Series:
        """Monthly returns for a static benchmark with one-way TC each month."""
        r = pd.Series(0.0, index=asset_returns.index)
        for asset, w in weights.items():
            if asset in asset_returns.columns:
                r += w * asset_returns[asset]
        # Approximate TC drag: monthly turnover ≈ 0 for static (drift-only),
        # but apply a small annual cost to be conservative
        r -= self.bench_tc / 12   # spread the round-trip cost over the year
        return r.dropna()

    def _dynamic_benchmark_returns(
        self,
        weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> pd.Series:
        """Monthly returns for a weight-series benchmark (e.g. inv-vol)."""
        w_lagged = weights.shift(1).reindex(asset_returns.index, method="ffill")
        r_list: list[float] = []
        dates: list[pd.Timestamp] = []
        for date in asset_returns.index:
            if date not in w_lagged.index:
                continue
            row_w = w_lagged.loc[date].fillna(0)
            ret = sum(
                row_w.get(a, 0.0) * float(asset_returns.loc[date, a])
                for a in self.assets
                if a in asset_returns.columns and not np.isnan(asset_returns.loc[date, a])
            )
            r_list.append(ret - self.bench_tc / 12)
            dates.append(date)
        return pd.Series(r_list, index=dates)

    def _equity_curve(self, returns: pd.Series) -> pd.DataFrame:
        cum = (1 + returns).cumprod()
        return pd.DataFrame(
            {
                "return": returns,
                "cumulative_return": cum,
                "equity_curve": cum * self.initial_capital,
            }
        )


# ── module-level helper ──────────────────────────────────────────────────

def _max_drawdown(cumulative: pd.Series) -> float:
    rolling_max = cumulative.cummax()
    dd = (cumulative - rolling_max) / rolling_max
    return float(dd.min())
