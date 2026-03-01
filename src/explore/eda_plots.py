from __future__ import annotations
"""
explore/eda_plots.py  —  E in OSEMN
--------------------------------------
Visual EDA: time-series grids, correlation heatmaps, distribution
plots, BIC elbow, and regime timeline overlaid on the S&P 500.

All styling, output paths, and NBER recession dates come from config.
"""

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── matplotlib style ─────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")


class EDAPlots:
    """Generate and save all EDA visualisations."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg["outputs"]["figures_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fmt: str = cfg["outputs"]["formats"]["figures"]
        self.dpi: int = cfg["outputs"]["dpi"]
        self.nber_recessions: list[dict] = cfg["exploration"]["nber_recessions"]

        # Colour palette for regimes
        self.regime_colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # green, amber, red

    # ── save helper ──────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.output_dir / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  [plot] saved → {path.name}")

    def _add_nber_shading(self, ax: plt.Axes, alpha: float = 0.18) -> None:
        """Grey bands for NBER recessions on a given axes."""
        for rec in self.nber_recessions:
            ax.axvspan(
                pd.Timestamp(rec["start"]),
                pd.Timestamp(rec["end"]),
                alpha=alpha,
                color="grey",
                label="_nber",
            )

    # ── individual plots ─────────────────────────────────────────────────

    def feature_timeseries(
        self, df: pd.DataFrame, n_cols: int = 3, title: str = "Macro Features (Z-scored)"
    ) -> None:
        """Grid of time-series subplots for every feature column."""
        n_feat = len(df.columns)
        n_rows = (n_feat + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), squeeze=False
        )
        axes_flat = axes.flatten()

        for i, col in enumerate(df.columns):
            ax = axes_flat[i]
            ax.plot(df.index, df[col], linewidth=0.9, color="steelblue")
            ax.axhline(0, color="red", linewidth=0.6, linestyle="--", alpha=0.7)
            self._add_nber_shading(ax)
            ax.set_title(col, fontsize=8, pad=3)
            ax.tick_params(labelsize=7)

        # Hide unused axes
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(title, fontsize=13, y=1.01)
        fig.tight_layout()
        self._save(fig, "01_feature_timeseries")

    def correlation_heatmap(self, corr_matrix: pd.DataFrame) -> None:
        """Lower-triangle correlation heatmap."""
        fig, ax = plt.subplots(figsize=(max(10, len(corr_matrix) * 0.65),
                                        max(8, len(corr_matrix) * 0.55)))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=len(corr_matrix) <= 20,
            fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.4,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(
            f"Feature Correlation Matrix ({self.cfg['exploration']['correlation_method'].capitalize()})",
            fontsize=13,
        )
        fig.tight_layout()
        self._save(fig, "02_correlation_heatmap")

    def distribution_grid(self, df: pd.DataFrame) -> None:
        """Histogram + KDE for each feature."""
        n_feat = len(df.columns)
        n_cols = 3
        n_rows = (n_feat + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), squeeze=False
        )
        axes_flat = axes.flatten()

        for i, col in enumerate(df.columns):
            ax = axes_flat[i]
            data = df[col].dropna()
            ax.hist(data, bins=40, density=True, alpha=0.6, color="steelblue", edgecolor="white")
            # KDE overlay
            from scipy.stats import gaussian_kde
            xs = np.linspace(data.min(), data.max(), 200)
            kde = gaussian_kde(data)
            ax.plot(xs, kde(xs), color="darkblue", linewidth=1.5)
            ax.set_title(f"{col}\nskew={data.skew():.2f} kurt={data.kurt():.2f}", fontsize=8)
            ax.tick_params(labelsize=7)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Feature Distributions (Z-scored)", fontsize=13, y=1.01)
        fig.tight_layout()
        self._save(fig, "03_feature_distributions")

    def bic_elbow(self, bic_scores: dict, aic_scores: dict | None = None) -> None:
        """BIC (and optionally AIC) vs. number of HMM states — the 'elbow' chart."""
        ks = sorted(bic_scores.keys())
        bics = [bic_scores[k] for k in ks]
        best_k = min(bic_scores, key=bic_scores.get)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, bics, "o-", color="steelblue", linewidth=2, markersize=9, label="BIC")

        if aic_scores:
            aics = [aic_scores[k] for k in ks]
            ax.plot(ks, aics, "s--", color="salmon", linewidth=2, markersize=8, label="AIC")

        ax.axvline(best_k, color="green", linestyle=":", linewidth=1.8,
                   label=f"Selected K={best_k}")
        ax.scatter([best_k], [bic_scores[best_k]], color="green", s=140, zorder=6)

        ax.set_xlabel("Number of Hidden States (K)", fontsize=12)
        ax.set_ylabel("Information Criterion (lower = better)", fontsize=12)
        ax.set_title("HMM State Selection — BIC / AIC Criterion", fontsize=13, fontweight="bold")
        ax.set_xticks(ks)
        ax.legend(fontsize=11)
        fig.tight_layout()
        self._save(fig, "04_hmm_bic_elbow")

    def regime_timeline(
        self,
        regime_probs: pd.DataFrame,
        spy_prices: pd.Series,
        regime_names: list[str],
    ) -> None:
        """
        Two-panel chart:
          Top    — S&P 500 (log scale) with NBER recession shading
          Bottom — stacked regime probability bands
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 9), sharex=True,
            gridspec_kw={"height_ratios": [1.5, 1]}
        )

        # Top: price
        aligned_spy = spy_prices.reindex(regime_probs.index, method="ffill")
        ax1.plot(aligned_spy.index, aligned_spy.values, color="navy", linewidth=1.3)
        self._add_nber_shading(ax1)
        ax1.set_yscale("log")
        ax1.set_ylabel("S&P 500 (log scale)", fontsize=11)
        ax1.set_title("Macro Regime Detection — Walk-Forward Out-of-Sample", fontsize=13,
                      fontweight="bold")

        # Bottom: regime probabilities as stacked area
        prob_cols = [c for c in regime_probs.columns if c.startswith("prob_")]
        data_stack = [regime_probs[c].fillna(0).values for c in prob_cols]
        colors = self.regime_colors[: len(prob_cols)]
        labels = regime_names[: len(prob_cols)]

        ax2.stackplot(
            regime_probs.index,
            data_stack,
            labels=labels,
            colors=colors,
            alpha=0.80,
        )
        self._add_nber_shading(ax2, alpha=0.10)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Regime Probability", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )
        ax2.legend(loc="upper left", fontsize=10)

        # NBER legend patch
        grey_patch = mpatches.Patch(color="grey", alpha=0.3, label="NBER Recession")
        ax1.legend(handles=[grey_patch], loc="upper left", fontsize=9)

        fig.tight_layout()
        self._save(fig, "05_regime_timeline")

    def feature_importance_per_regime(
        self, importance_df: pd.DataFrame, regime_names: list[str]
    ) -> None:
        """
        Horizontal bar chart of mean Z-scored feature value per regime.
        Shows which features discriminate each regime.
        """
        n_regimes = len(importance_df.columns)
        fig, axes = plt.subplots(
            1, n_regimes,
            figsize=(6 * n_regimes, max(6, len(importance_df) * 0.35)),
            sharey=True,
        )
        if n_regimes == 1:
            axes = [axes]

        for i, (col, ax) in enumerate(zip(importance_df.columns, axes)):
            vals = importance_df[col].sort_values()
            colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in vals]
            ax.barh(vals.index, vals.values, color=colors, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8)
            label = regime_names[i] if i < len(regime_names) else f"Regime {col}"
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.tick_params(labelsize=8)

        fig.suptitle("Mean Feature Value per Regime (Z-scored)", fontsize=13, y=1.01)
        fig.tight_layout()
        self._save(fig, "06_feature_importance_per_regime")
