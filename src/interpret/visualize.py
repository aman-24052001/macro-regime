from __future__ import annotations
"""
interpret/visualize.py  —  N in OSEMN
----------------------------------------
Production-quality charts for the interpret phase:
  1. Equity curves + drawdown panel
  2. Stacked allocation area chart
  3. HMM transition matrix heatmap
  4. Performance comparison bar chart
  5. Rolling Sharpe ribbon
  6. Crisis period table heatmap

All paths, formats, and DPI from config.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

# Palette: strategy is bold dark, benchmarks are muted
_PALETTE = {
    "Regime Strategy": "#2c3e50",
    "Buy & Hold SPY": "#e74c3c",
    "60/40 SPY/TLT": "#3498db",
    "Equal Weight": "#27ae60",
    "Inverse Vol Static": "#f39c12",
}

_ASSET_COLORS = {
    "SPY": "#e74c3c",
    "TLT": "#3498db",
    "GLD": "#f1c40f",
    "LQD": "#2ecc71",
    "HYG": "#e67e22",
    "BIL": "#95a5a6",
}


class Visualizer:
    """Save all interpret-phase visualisations."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg["outputs"]["figures_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fmt: str = cfg["outputs"]["formats"]["figures"]
        self.dpi: int = cfg["outputs"]["dpi"]
        self.crisis_periods: list[dict] = cfg["backtest"]["crisis_periods"]

    # ── save helper ──────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.output_dir / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  [viz] saved → {path.name}")

    def _shade_crises(self, ax: plt.Axes, alpha: float = 0.12) -> None:
        for c in self.crisis_periods:
            ax.axvspan(
                pd.Timestamp(c["start"]),
                pd.Timestamp(c["end"]),
                alpha=alpha,
                color="salmon",
                label="_crisis",
            )

    # ── charts ───────────────────────────────────────────────────────────

    def equity_and_drawdown(self, results: dict[str, pd.DataFrame]) -> None:
        """
        Two-panel chart:  top = equity curves (log), bottom = drawdowns.
        Strategy rendered with thicker line; benchmarks thinner.
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 11), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        for name, df in results.items():
            color = _PALETTE.get(name, None)
            is_strategy = name == "Regime Strategy"
            lw = 2.5 if is_strategy else 1.5
            alpha = 1.0 if is_strategy else 0.75
            zorder = 5 if is_strategy else 2

            ax1.plot(
                df.index, df["cumulative_return"],
                label=name, color=color, linewidth=lw, alpha=alpha, zorder=zorder,
            )
            dd = (df["cumulative_return"] - df["cumulative_return"].cummax()) \
                / df["cumulative_return"].cummax()
            if is_strategy:
                ax2.fill_between(df.index, dd, 0, alpha=0.25, color=color, zorder=3)
            ax2.plot(df.index, dd, color=color, linewidth=lw, alpha=alpha, zorder=zorder)

        self._shade_crises(ax1)
        self._shade_crises(ax2)

        ax1.set_yscale("log")
        ax1.set_ylabel("Cumulative Return (log scale)", fontsize=12)
        ax1.set_title(
            "Walk-Forward Backtest — Equity Curves (out-of-sample)",
            fontsize=14, fontweight="bold",
        )
        ax1.legend(loc="upper left", fontsize=10)
        ax1.axhline(1.0, color="black", lw=0.5, ls="--")

        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax2.set_ylabel("Drawdown", fontsize=12)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_title("Drawdown Analysis", fontsize=13, fontweight="bold")

        fig.tight_layout()
        self._save(fig, "07_equity_and_drawdown")

    def allocation_area(self, weights: pd.DataFrame) -> None:
        """Stacked area chart of dynamic asset weights over time."""
        fig, ax = plt.subplots(figsize=(16, 7))

        assets = [c for c in weights.columns]
        colors = [_ASSET_COLORS.get(a, "#cccccc") for a in assets]

        ax.stackplot(
            weights.index,
            [weights[a].fillna(0).values for a in assets],
            labels=assets,
            colors=colors,
            alpha=0.85,
        )
        self._shade_crises(ax, alpha=0.08)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylabel("Portfolio Weight", fontsize=12)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_title("Dynamic Asset Allocation Over Time", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10, ncol=3)
        fig.tight_layout()
        self._save(fig, "08_allocation_area")

    def transition_heatmap(
        self, trans_matrix: np.ndarray, regime_names: list[str]
    ) -> None:
        """HMM learned transition probability matrix."""
        fig, ax = plt.subplots(figsize=(7, 6))
        df = pd.DataFrame(trans_matrix, index=regime_names, columns=regime_names)
        sns.heatmap(
            df, ax=ax,
            annot=True, fmt=".3f",
            cmap="YlOrRd", vmin=0, vmax=1,
            linewidths=1.0,
            cbar_kws={"label": "Transition Probability"},
        )
        ax.set_title(
            "HMM Transition Matrix\n(Row = From, Col = To)",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("To Regime", fontsize=11)
        ax.set_ylabel("From Regime", fontsize=11)
        fig.tight_layout()
        self._save(fig, "09_transition_heatmap")

    def performance_bars(self, numeric_df: pd.DataFrame) -> None:
        """4-panel bar chart comparing key metrics across all strategies."""
        metrics = [
            ("cagr", "CAGR", True),
            ("sharpe_ratio", "Sharpe Ratio", False),
            ("max_drawdown", "Max Drawdown (abs)", True),
            ("calmar_ratio", "Calmar Ratio", False),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes_flat = axes.flatten()

        for i, (col, label, is_pct) in enumerate(metrics):
            ax = axes_flat[i]
            if col not in numeric_df.columns:
                continue
            vals = numeric_df[col].astype(float)
            if col == "max_drawdown":
                vals = vals.abs()
            colors = [
                "#2c3e50" if idx == "Regime Strategy" else "#bdc3c7"
                for idx in vals.index
            ]
            ax.bar(range(len(vals)), vals.values, color=colors, edgecolor="white", width=0.6)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(vals.index, rotation=25, ha="right", fontsize=9)
            ax.set_title(label, fontsize=12, fontweight="bold")
            if is_pct:
                ax.yaxis.set_major_formatter(
                    mticker.PercentFormatter(xmax=1.0, decimals=1)
                )

        fig.suptitle("Performance Comparison — All Strategies", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "10_performance_comparison")

    def rolling_sharpe(
        self,
        results: dict[str, pd.DataFrame],
        window: int = 36,
        risk_free_monthly: float = 0.0,
    ) -> None:
        """Rolling Sharpe ratio ribbon for strategy vs benchmarks."""
        fig, ax = plt.subplots(figsize=(16, 6))

        for name, df in results.items():
            r = df["return"] - risk_free_monthly
            rs = (
                r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(12)
            )
            color = _PALETTE.get(name, None)
            lw = 2.5 if name == "Regime Strategy" else 1.3
            ax.plot(rs.index, rs, label=name, color=color, linewidth=lw,
                    alpha=1.0 if name == "Regime Strategy" else 0.7)

        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.axhline(1, color="green", lw=0.7, ls=":", alpha=0.7)
        self._shade_crises(ax)
        ax.set_ylabel(f"Rolling {window}-Month Sharpe", fontsize=12)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_title(f"Rolling Sharpe Ratio ({window}M window)", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        fig.tight_layout()
        self._save(fig, "11_rolling_sharpe")

    def overlay_signal(
        self,
        ticker: str,
        signal_df: "pd.DataFrame",
        key_events: list[dict] | None = None,
    ) -> None:
        """
        3-panel chart for stock overlay signals.
          Top    : Composite score over time (with fill)
          Middle : Recommended weight (%)
          Bottom : Annualised volatility

        Args:
            ticker      : e.g. 'NVDA'
            signal_df   : output from CompositeSignal.compute()
            key_events  : list of {date, label, color} for event annotations
        """
        if signal_df.empty:
            return

        fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
        ax_score, ax_weight, ax_vol = axes

        dates = signal_df.index

        # ── Panel 1: Composite score ───────────────────────────────────
        score = signal_df["composite_score"].fillna(0)
        ax_score.fill_between(dates, score, 0,
                              where=(score >= 0), alpha=0.3, color="#2ecc71", label="Positive")
        ax_score.fill_between(dates, score, 0,
                              where=(score < 0), alpha=0.3, color="#e74c3c", label="Negative")
        ax_score.plot(dates, score, color="#2c3e50", lw=1.5)
        ax_score.axhline(0, color="black", lw=0.7, ls="--")
        ax_score.set_ylabel("Composite Score", fontsize=11)
        ax_score.set_title(
            f"{ticker} — Multi-Signal Adaptive Composite (5 signals)",
            fontsize=13, fontweight="bold"
        )
        ax_score.legend(loc="upper left", fontsize=9)
        ax_score.set_ylim(-1.2, 1.2)

        # ── Panel 2: Recommended weight ───────────────────────────────
        weight = signal_df["weight"].fillna(0) * 100
        ax_weight.fill_between(dates, weight, 0, alpha=0.5, color="#3498db")
        ax_weight.plot(dates, weight, color="#2980b9", lw=1.5)
        ax_weight.axhline(25, color="red", lw=0.8, ls=":", alpha=0.7, label="Max weight (25%)")
        ax_weight.set_ylabel("Recommended Weight (%)", fontsize=11)
        ax_weight.set_ylim(0, 30)
        ax_weight.legend(loc="upper left", fontsize=9)

        # ── Panel 3: Annualised vol ────────────────────────────────────
        vol = signal_df["sigma_annual"].ffill() * 100
        ax_vol.plot(dates, vol, color="#f39c12", lw=1.5)
        ax_vol.set_ylabel("Annualised Vol (%)", fontsize=11)
        ax_vol.set_xlabel("Date", fontsize=11)

        # ── Key event annotations ─────────────────────────────────────
        for ev in (key_events or []):
            dt = pd.Timestamp(ev["date"])
            col = ev.get("color", "#999")
            for ax in axes:
                ax.axvline(dt, color=col, lw=1.0, ls="--", alpha=0.7)
            ax_score.annotate(
                ev["label"],
                xy=(dt, 0.9), fontsize=7.5, color=col,
                rotation=85, ha="right", va="top",
            )

        fig.tight_layout()
        self._save(fig, f"12_overlay_{ticker.lower()}")
