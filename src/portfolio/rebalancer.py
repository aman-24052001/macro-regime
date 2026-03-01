from __future__ import annotations
"""
portfolio/rebalancer.py  —  N in OSEMN
-----------------------------------------
Simulates monthly portfolio rebalancing with:
  • Threshold bands — only trade if any asset deviates > threshold_pct
  • Realistic transaction costs — commission + slippage from config
  • Weight drift tracking — weights drift between rebalance dates
  • Full output DataFrame — returns, costs, turnover, actual weights per month

One-period lag contract:
  Weights computed from regime_probs at T are APPLIED to returns at T+1.
  This prevents any look-ahead bias in the simulation.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Rebalancer:
    """Simulate portfolio with threshold rebalancing and transaction costs."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        reb = cfg["portfolio"]["rebalancing"]

        self.threshold: float = reb["threshold_pct"] / 100.0   # convert bps → fraction
        tc_bps = reb["transaction_cost_bps"] + reb["slippage_bps"]
        self.tc_per_unit: float = tc_bps / 10_000.0           # bps → fraction (one-way)

        self.assets: list[str] = list(cfg["data"]["market"]["etfs"].keys())

    # ── public ───────────────────────────────────────────────────────────

    def simulate(
        self,
        target_weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the full simulation.

        Args:
            target_weights : weights at time T  (from allocator, lag-1 applied here)
            asset_returns  : monthly returns  (date × asset)

        Returns:
            DataFrame indexed by date with columns:
              portfolio_return, gross_return, tc_drag, turnover,
              rebalanced, w_{asset} for each asset.

        One-period lag: weights at T are applied to returns at T+1.
        We iterate over *returns* dates; the weight at each date is
        the target weight from the PREVIOUS month.
        """
        # Align indices — apply weights[T] to returns[T+1]
        common_dates = asset_returns.index.intersection(
            target_weights.shift(1, freq="BME").index
            if hasattr(target_weights.index, "freq")
            else target_weights.index
        )

        # Simpler: use shifted weights directly
        weights_lagged = target_weights.shift(1)   # weight at T → applies at T+1
        weights_lagged = weights_lagged.reindex(asset_returns.index, method="ffill")

        # Initial equal-weight position
        n_assets = len(self.assets)
        w_current = {a: 1.0 / n_assets for a in self.assets}

        results: list[dict] = []
        dates_used: list[pd.Timestamp] = []

        for date in asset_returns.index:
            if date not in weights_lagged.index:
                continue

            target_row = weights_lagged.loc[date]
            if target_row.isna().all():
                continue
            w_target = {a: float(target_row.get(a, 0.0)) for a in self.assets}

            # Decide whether to rebalance
            max_dev = max(
                abs(w_target.get(a, 0.0) - w_current.get(a, 0.0)) for a in self.assets
            )
            rebalanced = max_dev >= self.threshold

            if rebalanced:
                trades = {
                    a: w_target.get(a, 0.0) - w_current.get(a, 0.0) for a in self.assets
                }
                turnover = sum(abs(v) for v in trades.values()) / 2.0   # one-way
                tc_drag = turnover * self.tc_per_unit * 2               # round-trip
                w_applied = w_target
            else:
                trades = {}
                turnover = 0.0
                tc_drag = 0.0
                w_applied = w_current

            # Gross return this period
            gross = sum(
                w_applied.get(a, 0.0) * float(asset_returns.loc[date, a])
                for a in self.assets
                if a in asset_returns.columns and not np.isnan(asset_returns.loc[date, a])
            )
            net = gross - tc_drag

            # Alert on drawdown breach
            dd_limit = self.cfg["portfolio"]["risk_management"]["max_drawdown_alert"]

            row: dict = {
                "gross_return": gross,
                "tc_drag": tc_drag,
                "portfolio_return": net,
                "turnover": turnover,
                "rebalanced": int(rebalanced),
            }
            row.update({f"w_{a}": w_applied.get(a, 0.0) for a in self.assets})
            results.append(row)
            dates_used.append(date)

            # Update weights for drift (end-of-period weights)
            new_vals = {
                a: w_applied.get(a, 0.0) * (1.0 + float(asset_returns.loc[date, a]))
                for a in self.assets
                if a in asset_returns.columns and not np.isnan(asset_returns.loc[date, a])
            }
            total = sum(new_vals.values())
            w_current = {a: v / total for a, v in new_vals.items()} if total > 1e-9 else w_applied

        df = pd.DataFrame(results, index=dates_used)

        # Summary stats
        avg_tc = df["tc_drag"].mean() * 12 * 100
        avg_turn = df["turnover"].mean() * 100
        logger.info(
            f"Simulation complete: {len(df)} months  "
            f"avg annual TC drag={avg_tc:.2f}bps  "
            f"avg monthly turnover={avg_turn:.1f}%  "
            f"rebalanced {df['rebalanced'].sum()} times"
        )
        return df
