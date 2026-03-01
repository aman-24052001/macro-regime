from __future__ import annotations
"""
model/bootstrap.py  —  Block Bootstrap Confidence Intervals
------------------------------------------------------------
Tests statistical significance of the strategy's Sharpe ratio vs SPY.

Method: stationary block bootstrap (Politis & Romano 1994)
  - Block length: 12 months (preserves annual seasonality)
  - Simulations: 1000 resamples
  - Reports: 95% CI for Sharpe, CAGR, MaxDD; p-value Sharpe > SPY

Outputs
-------
  outputs/reports/bootstrap_results.csv
  outputs/figures/14_bootstrap_ci.png
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MONTHS_PER_YEAR = 12

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")


class BlockBootstrap:
    """Stationary block bootstrap CI for portfolio performance metrics."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.out_dir  = Path(cfg["outputs"]["reports_dir"])
        self.fig_dir  = Path(cfg["outputs"]["figures_dir"])
        self.proc_dir = Path(cfg["outputs"]["processed_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    # ── public ───────────────────────────────────────────────────────────

    def run(
        self,
        n_boot: int = 1000,
        block_length: int = 12,
        spy_sharpe: float = 0.760,
        spy_cagr: float = 0.1055,
    ) -> dict:
        """
        Run block bootstrap and return summary dict.

        Parameters
        ----------
        n_boot       : number of bootstrap simulations
        block_length : block size in months (12 = annual blocks)
        spy_sharpe   : SPY OOS Sharpe to test against
        spy_cagr     : SPY OOS CAGR to test against
        """
        # Load OOS portfolio returns
        strategy_ret, spy_ret = self._load_returns()
        n = len(strategy_ret)

        logger.info("=" * 58)
        logger.info("  Block Bootstrap Confidence Intervals")
        logger.info(f"  n={n} months  block_len={block_length}  sims={n_boot}")
        logger.info("=" * 58)

        # Bootstrap
        boot_sharpe = np.empty(n_boot)
        boot_cagr   = np.empty(n_boot)
        boot_maxdd  = np.empty(n_boot)

        rng = np.random.default_rng(42)
        for i in range(n_boot):
            sample = self._block_resample(strategy_ret.values, block_length, rng)
            boot_sharpe[i] = self._sharpe(sample)
            boot_cagr[i]   = self._cagr(sample)
            boot_maxdd[i]  = self._max_dd(sample)

        # Point estimates
        pt_sharpe = self._sharpe(strategy_ret.values)
        pt_cagr   = self._cagr(strategy_ret.values)
        pt_maxdd  = self._max_dd(strategy_ret.values)

        # CIs and p-values
        sharpe_ci  = np.percentile(boot_sharpe, [2.5, 97.5])
        cagr_ci    = np.percentile(boot_cagr,   [2.5, 97.5])
        maxdd_ci   = np.percentile(boot_maxdd,  [2.5, 97.5])

        p_sharpe   = (boot_sharpe > spy_sharpe).mean()
        p_cagr     = (boot_cagr   > spy_cagr).mean()

        results = {
            "n_months":          n,
            "n_boot":            n_boot,
            "block_length":      block_length,
            "sharpe_point":      round(pt_sharpe, 4),
            "sharpe_ci_lo":      round(sharpe_ci[0], 4),
            "sharpe_ci_hi":      round(sharpe_ci[1], 4),
            "sharpe_pval_vs_spy": round(p_sharpe, 4),
            "cagr_point":        round(pt_cagr, 4),
            "cagr_ci_lo":        round(cagr_ci[0], 4),
            "cagr_ci_hi":        round(cagr_ci[1], 4),
            "cagr_pval_vs_spy":  round(p_cagr, 4),
            "maxdd_point":       round(pt_maxdd, 4),
            "maxdd_ci_lo":       round(maxdd_ci[0], 4),
            "maxdd_ci_hi":       round(maxdd_ci[1], 4),
            "spy_sharpe":        spy_sharpe,
            "spy_cagr":          spy_cagr,
        }

        # Save CSV
        pd.DataFrame([results]).to_csv(
            self.out_dir / "bootstrap_results.csv", index=False
        )

        # Log
        logger.info(
            f"\n  Sharpe  : {pt_sharpe:.3f}  95% CI [{sharpe_ci[0]:.3f}, {sharpe_ci[1]:.3f}]"
            f"  p(>SPY {spy_sharpe:.3f}) = {p_sharpe:.3f}"
        )
        logger.info(
            f"  CAGR    : {pt_cagr:.1%}  95% CI [{cagr_ci[0]:.1%}, {cagr_ci[1]:.1%}]"
            f"  p(>SPY {spy_cagr:.1%}) = {p_cagr:.3f}"
        )
        logger.info(
            f"  Max DD  : {pt_maxdd:.1%}  95% CI [{maxdd_ci[0]:.1%}, {maxdd_ci[1]:.1%}]"
        )
        logger.info(f"  Saved → bootstrap_results.csv")

        self._plot(boot_sharpe, pt_sharpe, spy_sharpe,
                   boot_cagr,   pt_cagr,   spy_cagr,
                   boot_maxdd,  pt_maxdd)

        return results

    # ── private ──────────────────────────────────────────────────────────

    def _load_returns(self) -> tuple[pd.Series, pd.Series]:
        """Load OOS portfolio returns and SPY returns."""
        # Strategy returns — saved by interpret phase
        ret_path = self.out_dir / "portfolio_returns.csv"
        if not ret_path.exists():
            raise FileNotFoundError(
                f"portfolio_returns.csv not found at {ret_path}. "
                "Run --phase interpret first."
            )
        bt = pd.read_csv(ret_path, index_col=0, parse_dates=True)
        strategy = bt["portfolio_return"].dropna()

        # SPY — from asset returns
        ar_path = self.proc_dir / "asset_returns.csv"
        ar = pd.read_csv(ar_path, index_col=0, parse_dates=True)
        spy = ar["SPY"].reindex(strategy.index).fillna(0.0)

        return strategy, spy

    def _block_resample(
        self,
        returns: np.ndarray,
        block_length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Stationary block bootstrap resample."""
        n = len(returns)
        result = np.empty(n)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            take  = min(block_length, n - i)
            block = np.array([returns[(start + j) % n] for j in range(take)])
            result[i:i + take] = block
            i += take
        return result

    def _sharpe(self, r: np.ndarray, rf_annual: float = 0.02) -> float:
        rf_monthly = (1 + rf_annual) ** (1 / MONTHS_PER_YEAR) - 1
        excess = r - rf_monthly
        if excess.std() < 1e-9:
            return float("nan")
        return float(excess.mean() / excess.std() * np.sqrt(MONTHS_PER_YEAR))

    def _cagr(self, r: np.ndarray) -> float:
        n = len(r)
        if n == 0:
            return float("nan")
        return float((1 + r).prod() ** (MONTHS_PER_YEAR / n) - 1)

    def _max_dd(self, r: np.ndarray) -> float:
        cum = (1 + r).cumprod()
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        return float(dd.min())

    def _plot(
        self,
        boot_sharpe: np.ndarray,
        pt_sharpe: float,
        spy_sharpe: float,
        boot_cagr: np.ndarray,
        pt_cagr: float,
        spy_cagr: float,
        boot_maxdd: np.ndarray,
        pt_maxdd: float,
    ) -> None:
        """3-panel bootstrap distribution figure."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            "Block Bootstrap (n=1000, block=12m) — Strategy vs SPY",
            fontsize=13, fontweight="bold", y=1.01
        )

        ci95_lo = np.percentile(boot_sharpe, 2.5)
        ci95_hi = np.percentile(boot_sharpe, 97.5)

        # Panel 1: Sharpe histogram
        ax = axes[0]
        ax.hist(boot_sharpe, bins=50, color="#4f7cff", alpha=0.7, edgecolor="none")
        ax.axvline(pt_sharpe,  color="#34c97b", lw=2, label=f"Strategy {pt_sharpe:.3f}")
        ax.axvline(spy_sharpe, color="#e05555", lw=2, ls="--", label=f"SPY {spy_sharpe:.3f}")
        ax.axvspan(ci95_lo, ci95_hi, alpha=0.15, color="#4f7cff", label="95% CI")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Sharpe  [{ci95_lo:.3f}, {ci95_hi:.3f}]")
        ax.legend(fontsize=9)

        # Panel 2: CAGR histogram
        ax = axes[1]
        ax.hist(boot_cagr * 100, bins=50, color="#34c97b", alpha=0.7, edgecolor="none")
        ax.axvline(pt_cagr * 100,  color="#34c97b", lw=2, label=f"Strategy {pt_cagr:.1%}")
        ax.axvline(spy_cagr * 100, color="#e05555", lw=2, ls="--", label=f"SPY {spy_cagr:.1%}")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ci_lo2 = np.percentile(boot_cagr, 2.5) * 100
        ci_hi2 = np.percentile(boot_cagr, 97.5) * 100
        ax.axvspan(ci_lo2, ci_hi2, alpha=0.15, color="#34c97b")
        ax.set_xlabel("CAGR (%)")
        ax.set_title(f"CAGR  [{ci_lo2:.1f}%, {ci_hi2:.1f}%]")
        ax.legend(fontsize=9)

        # Panel 3: Max Drawdown histogram
        ax = axes[2]
        ax.hist(boot_maxdd * 100, bins=50, color="#e05555", alpha=0.7, edgecolor="none")
        ax.axvline(pt_maxdd * 100, color="#e05555", lw=2, label=f"Strategy {pt_maxdd:.1%}")
        ci_lo3 = np.percentile(boot_maxdd, 2.5) * 100
        ci_hi3 = np.percentile(boot_maxdd, 97.5) * 100
        ax.axvspan(ci_lo3, ci_hi3, alpha=0.15, color="#e05555")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_title(f"Max DD  [{ci_lo3:.1f}%, {ci_hi3:.1f}%]")
        ax.legend(fontsize=9)

        plt.tight_layout()
        out_path = self.fig_dir / "14_bootstrap_ci.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  [bootstrap] saved → {out_path.name}")
