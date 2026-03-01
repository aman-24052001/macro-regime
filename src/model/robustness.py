from __future__ import annotations
"""
model/robustness.py  —  Sensitivity & Robustness Analysis
----------------------------------------------------------
Tests model stability across a parameter grid:

  HMM axis   : K ∈ {2, 3, 4}  × initial_train_years ∈ {10, 15, 20}
  Portfolio  : TC ∈ {0, 5, 15, 30} bps × threshold ∈ {0%, 2%, 5%}

HMM walk-forward is run once per (K, train_years) pair — 9 combinations,
each with n_fits=3 (fast, for sensitivity only — not production).
TC and threshold are re-applied to cached regime_probs without retraining.

Total grid: 9 HMM runs × 12 portfolio combos = 108 result rows.

Outputs
-------
  outputs/reports/robustness_grid.csv
  outputs/figures/13_robustness_grid.png
"""

import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.model.hmm_trainer import HMMTrainer
from src.portfolio.allocator import MacroAllocator
from src.portfolio.rebalancer import Rebalancer
from src.interpret.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")


class RobustnessGrid:
    """Run a full sensitivity analysis across the parameter grid."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg["outputs"]["reports_dir"])
        self.fig_dir = Path(cfg["outputs"]["figures_dir"])
        self.proc_dir = Path(cfg["outputs"]["processed_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    # ── public ───────────────────────────────────────────────────────────

    def run(
        self,
        k_values: list[int]         = [2, 3, 4],
        train_years_values: list[int] = [10, 15, 20],
        tc_bps_values: list[float]   = [0.0, 5.0, 15.0, 30.0],
        threshold_pct_values: list[float] = [0.0, 2.0, 5.0],
        n_fits: int = 3,
        oos_start: str | None = None,
    ) -> pd.DataFrame:
        """
        Run the full robustness grid.

        Returns
        -------
        DataFrame with one row per (K, train_years, tc_bps, threshold_pct)
        and columns: sharpe, cagr, max_dd, sortino, calmar, volatility.
        """
        if oos_start is None:
            oos_start = self.cfg["backtest"]["start_date"]

        # Load data from cache (built during scrub phase)
        features = self._load_features()
        asset_returns = self._load_asset_returns()

        logger.info("=" * 58)
        logger.info("  Robustness Grid")
        logger.info(f"  K={k_values}  train_yrs={train_years_values}")
        logger.info(f"  TC={tc_bps_values}bps  threshold={threshold_pct_values}%")
        logger.info(f"  n_fits={n_fits} (fast mode — sensitivity only)")
        logger.info("=" * 58)

        # ── Step 1: HMM walk-forward for each (K, train_years) ──
        hmm_cache: dict[tuple, dict] = {}
        n_hmm = len(k_values) * len(train_years_values)
        run_num = 0

        for k in k_values:
            for train_yrs in train_years_values:
                run_num += 1
                key = (k, train_yrs)
                logger.info(
                    f"\n[HMM {run_num}/{n_hmm}] K={k}  train_years={train_yrs} ..."
                )
                cfg_mod = self._modify_cfg(k=k, train_years=train_yrs, n_fits=n_fits)
                trainer = HMMTrainer(cfg_mod)

                try:
                    regime_probs = trainer.walk_forward(features, k, oos_start)
                    full_model   = trainer.fit_full(
                        features, k, name=f"hmm_rob_K{k}_y{train_yrs}"
                    )
                    feat_names = list(
                        trainer._select_features(features).columns
                    )
                    regime_names = self._label_for_k(
                        trainer, full_model, feat_names, k
                    )
                    hmm_cache[key] = {
                        "regime_probs": regime_probs,
                        "regime_names": regime_names,
                    }
                    logger.info(
                        f"  → {len(regime_probs)} OOS months  "
                        f"labels={regime_names}"
                    )
                except Exception as exc:
                    logger.error(f"  [K={k} y={train_yrs}] HMM failed: {exc}")

        # ── Step 2: Portfolio sweep over TC × threshold ──
        results: list[dict] = []
        n_port = len(hmm_cache) * len(tc_bps_values) * len(threshold_pct_values)
        port_num = 0

        for (k, train_yrs), hmm_data in hmm_cache.items():
            for tc in tc_bps_values:
                for thresh in threshold_pct_values:
                    port_num += 1
                    cfg_mod = self._modify_cfg(
                        k=k, train_years=train_yrs,
                        tc_bps=tc, threshold_pct=thresh,
                    )
                    try:
                        metrics = self._run_portfolio(
                            cfg_mod,
                            hmm_data["regime_probs"],
                            hmm_data["regime_names"],
                            asset_returns,
                        )
                        results.append({
                            "K":              k,
                            "train_years":    train_yrs,
                            "tc_bps":         tc,
                            "threshold_pct":  thresh,
                            **metrics,
                        })
                    except Exception as exc:
                        logger.error(
                            f"  [K={k} y={train_yrs} tc={tc} thr={thresh}] "
                            f"portfolio failed: {exc}"
                        )

        logger.info(
            f"\nGrid complete: {len(results)}/{n_port} combos succeeded."
        )

        df = pd.DataFrame(results)
        out_csv = self.out_dir / "robustness_grid.csv"
        df.to_csv(out_csv, index=False)
        logger.info(f"  [robustness] saved → {out_csv.name}")

        if df.empty:
            logger.warning("  [robustness] No results to plot — skipping figure.")
        else:
            self._plot(df)
        return df

    # ── private helpers ───────────────────────────────────────────────────

    def _load_features(self) -> pd.DataFrame:
        path = self.proc_dir / "features_full.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"features_full.csv not found at {path}. "
                "Run --phase scrub first."
            )
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def _load_asset_returns(self) -> pd.DataFrame:
        path = self.proc_dir / "asset_returns.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"asset_returns.csv not found at {path}. "
                "Run --phase scrub first."
            )
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def _modify_cfg(
        self,
        k: int,
        train_years: int = 15,
        n_fits: int = 3,
        tc_bps: float | None = None,
        threshold_pct: float | None = None,
    ) -> dict:
        """Deep-copy cfg and override specified parameters."""
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["hmm"]["force_k"] = k
        cfg["model"]["walk_forward"]["initial_train_years"] = train_years
        cfg["model"]["hmm"]["walk_forward_n_fits"] = n_fits
        if tc_bps is not None:
            cfg["portfolio"]["rebalancing"]["transaction_cost_bps"] = tc_bps
        if threshold_pct is not None:
            cfg["portfolio"]["rebalancing"]["threshold_pct"] = threshold_pct
        return cfg

    def _label_for_k(
        self,
        trainer: HMMTrainer,
        model,
        feature_names: list[str],
        k: int,
    ) -> list[str]:
        """
        Assign portfolio-compatible regime names for any K.

        Strategy:
          - Sort states by yield_curve_10y2y emission mean (ascending)
          - Lowest  → 'contraction'
          - Highest → 'expansion'
          - Middle  → 'stagnation'  (all middle states share this label)

        For K=2: no stagnation state; allocator gets {expansion, contraction}.
        For K=4: two middle states both map to stagnation (probability doubles
                 on the stagnation portfolio; allocator normalises correctly).
        """
        yc_idx = next(
            (i for i, f in enumerate(feature_names) if "yield_curve_10y2y" in f),
            None,
        )
        if yc_idx is None:
            return [f"regime_{i}" for i in range(k)]

        yc_vals = model.means_[:, yc_idx]
        order = np.argsort(yc_vals)   # ascending → index 0 = most negative YC

        labels = ["stagnation"] * k   # default everything to stagnation
        labels[order[0]]  = "contraction"
        labels[order[-1]] = "expansion"
        return labels

    def _run_portfolio(
        self,
        cfg: dict,
        regime_probs: pd.DataFrame,
        regime_names: list[str],
        asset_returns: pd.DataFrame,
    ) -> dict:
        """Allocate → rebalance → compute metrics. Returns metric dict."""
        allocator   = MacroAllocator(cfg)
        weights     = allocator.compute_weights(regime_probs, regime_names)

        rebalancer  = Rebalancer(cfg)
        portfolio   = rebalancer.simulate(weights, asset_returns)

        metrics_calc = PerformanceMetrics(cfg)
        m = metrics_calc.compute(portfolio["portfolio_return"])

        return {
            "sharpe":     round(m.get("sharpe_ratio",  float("nan")), 4),
            "sortino":    round(m.get("sortino_ratio", float("nan")), 4),
            "calmar":     round(m.get("calmar_ratio",  float("nan")), 4),
            "cagr":       round(m.get("cagr",          float("nan")), 4),
            "max_dd":     round(m.get("max_drawdown",  float("nan")), 4),
            "volatility": round(m.get("volatility",    float("nan")), 4),
        }

    # ── plots ─────────────────────────────────────────────────────────────

    def _plot(self, df: pd.DataFrame) -> None:
        """Generate the 3-panel robustness figure."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # ── Panel 1: Sharpe heatmap  K × train_years (baseline TC=5, thr=2) ──
        ax = axes[0]
        base = df[(df["tc_bps"] == 5.0) & (df["threshold_pct"] == 2.0)]
        if not base.empty:
            pivot = base.pivot(index="train_years", columns="K", values="sharpe")
            im = ax.imshow(
                pivot.values, cmap="RdYlGn",
                vmin=0.5, vmax=1.1, aspect="auto",
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"K={c}" for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{y}yr" for y in pivot.index])
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                                fontsize=11, fontweight="bold",
                                color="white" if (v < 0.65 or v > 1.0) else "black")
            plt.colorbar(im, ax=ax, label="Sharpe Ratio")
            ax.set_title(
                "Sharpe Ratio — K × Training Years\n(TC=5bps, threshold=2%)",
                fontsize=12, fontweight="bold",
            )
            ax.set_xlabel("HMM States (K)")
            ax.set_ylabel("Initial Training Window")

        # ── Panel 2: TC sensitivity (K=3, train=15) ──
        ax = axes[1]
        base_tc = df[(df["K"] == 3) & (df["train_years"] == 15) & (df["threshold_pct"] == 2.0)]
        if not base_tc.empty:
            ax.plot(
                base_tc["tc_bps"], base_tc["sharpe"],
                "o-", color="#4f7cff", linewidth=2.5, markersize=8,
                label="Sharpe",
            )
            ax.plot(
                base_tc["tc_bps"], base_tc["calmar"],
                "s--", color="#34c97b", linewidth=1.8, markersize=7,
                label="Calmar",
            )
            ax.axhline(
                0.760, color="#e05555", linewidth=1.2, linestyle=":",
                label="SPY Sharpe (0.760)",
            )
            ax.set_xlabel("One-Way Transaction Cost (bps)")
            ax.set_ylabel("Ratio")
            ax.set_title(
                "TC Sensitivity\n(K=3, train=15yr, threshold=2%)",
                fontsize=12, fontweight="bold",
            )
            ax.legend(fontsize=9)
            ax.set_xticks(base_tc["tc_bps"].values)

        # ── Panel 3: Rebalance threshold sensitivity (K=3, train=15, TC=5) ──
        ax = axes[2]
        base_thr = df[(df["K"] == 3) & (df["train_years"] == 15) & (df["tc_bps"] == 5.0)]
        if not base_thr.empty:
            x = np.arange(len(base_thr))
            width = 0.35
            bars1 = ax.bar(
                x - width / 2, base_thr["sharpe"].values,
                width, label="Sharpe", color="#4f7cff", alpha=0.85, edgecolor="white",
            )
            bars2 = ax.bar(
                x + width / 2, base_thr["cagr"].values * 10,  # scale CAGR for visibility
                width, label="CAGR × 10", color="#34c97b", alpha=0.75, edgecolor="white",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([f"{t:.0f}%" for t in base_thr["threshold_pct"].values])
            ax.set_xlabel("Rebalance Threshold")
            ax.set_ylabel("Metric Value")
            ax.set_title(
                "Rebalance Threshold Sensitivity\n(K=3, train=15yr, TC=5bps)",
                fontsize=12, fontweight="bold",
            )
            ax.legend(fontsize=9)

            # Label bars
            for bar in bars1:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9,
                )

        fig.suptitle(
            "Robustness Sensitivity Analysis — Parameter Grid",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()
        out_path = self.fig_dir / "13_robustness_grid.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  [robustness] figure saved → {out_path.name}")
