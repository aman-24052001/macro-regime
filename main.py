from __future__ import annotations
#!/usr/bin/env python3
"""
main.py — CLI entry point for the full OSEMN pipeline
======================================================
Adaptive Macro Regime Detection & Dynamic Factor Rotation Strategy

Usage:
    python main.py                          # full pipeline, all defaults
    python main.py --refresh                # re-download all data
    python main.py --phase obtain           # run only the Obtain phase
    python main.py --config path/to/cfg.yaml
    python main.py --log-level DEBUG

OSEMN flow:
    O — Obtain   : download FRED macro + yfinance market data
    S — Scrub    : preprocess, feature engineer
    E — Explore  : EDA stats + plots, BIC state selection (data-driven)
    M — Model    : walk-forward HMM training at chosen K
    N — iNterpret: portfolio construction, backtest, metrics, visualisations
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── imports ──────────────────────────────────────────────────────────────

from src.config_loader import load_config
from src.obtain.fred_loader import FREDLoader
from src.obtain.market_loader import MarketLoader
from src.scrub.preprocessor import Preprocessor
from src.scrub.feature_engineer import FeatureEngineer
from src.explore.eda_stats import EDAStats
from src.explore.eda_plots import EDAPlots
from src.model.hmm_selector import HMMSelector
from src.model.hmm_trainer import HMMTrainer
from src.model.composite_signal import CompositeSignal
from src.model.robustness import RobustnessGrid
from src.model.bootstrap import BlockBootstrap
from src.model.garch_model import GARCHModel
from src.portfolio.allocator import MacroAllocator
from src.portfolio.rebalancer import Rebalancer
from src.interpret.backtest import Backtester
from src.interpret.metrics import PerformanceMetrics
from src.interpret.visualize import Visualizer


# ── setup ────────────────────────────────────────────────────────────────

def setup_logging(level: str) -> None:
    # Use UTF-8 on all handlers so Windows cp1252 terminals don't choke
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.stream.reconfigure(encoding="utf-8", errors="replace")
    file_handler = logging.FileHandler("outputs/pipeline.log", mode="a", encoding="utf-8")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[stream_handler, file_handler],
    )


# ── phase functions ──────────────────────────────────────────────────────

def phase_obtain(cfg: dict, refresh: bool) -> tuple:
    """O — Download FRED macro series and market price data."""
    log = logging.getLogger("obtain")
    log.info("=" * 55)
    log.info("  Phase O — OBTAIN")
    log.info("=" * 55)

    fred_data = FREDLoader(cfg).load(force_refresh=refresh)
    market_data = MarketLoader(cfg).load(force_refresh=refresh)

    log.info(
        f"Obtained: {len(fred_data)} FRED series, "
        f"{len(market_data)} market instruments."
    )
    return fred_data, market_data


def phase_scrub(cfg: dict, fred_data: dict, market_data: dict) -> tuple:
    """S — Clean, align, normalise, engineer features."""
    log = logging.getLogger("scrub")
    log.info("=" * 55)
    log.info("  Phase S — SCRUB")
    log.info("=" * 55)

    preprocessor = Preprocessor(cfg)
    df_zscored, df_raw = preprocessor.build_feature_matrix(fred_data)

    assets = list(cfg["data"]["market"]["etfs"].keys())
    asset_returns = preprocessor.build_returns_matrix(market_data, assets)

    fe = FeatureEngineer(cfg)
    features = fe.engineer(df_raw, df_zscored)

    log.info(f"Feature matrix: {features.shape}  |  Returns: {asset_returns.shape}")
    return features, df_raw, asset_returns


def phase_explore(
    cfg: dict,
    features,
    df_raw,
    asset_returns,
) -> dict:
    """
    E — Statistical EDA + data-driven BIC selection.

    Returns:
        dict with keys: best_k, bic_scores, aic_scores, bic_recommendation,
        stationarity, correlation, selection_result
    """
    log = logging.getLogger("explore")
    log.info("=" * 55)
    log.info("  Phase E — EXPLORE")
    log.info("=" * 55)

    stats = EDAStats(cfg)
    plots = EDAPlots(cfg)

    # Statistical reports
    stationarity = stats.stationarity_report(features)
    log.info(f"\nStationarity (top 10 non-stationary):\n"
             f"{stationarity.head(10).to_string(index=False)}")

    corr_matrix = stats.correlation_matrix(features)
    high_corr = stats.high_correlation_pairs(corr_matrix)
    if not high_corr.empty:
        log.info(f"\nHigh-correlation pairs:\n{high_corr.to_string(index=False)}")

    desc = stats.descriptive_stats(features)
    log.info(f"\nDescriptive stats shape: {desc.shape}")

    # Visual EDA
    plots.feature_timeseries(features)
    plots.correlation_heatmap(corr_matrix)
    plots.distribution_grid(features)

    # BIC — data-driven state selection
    log.info("\nRunning BIC state selection (this may take a few minutes) ...")
    oos_start = cfg["exploration"]["oos_start"]

    # Apply the same feature subset that HMMTrainer uses (features_for_hmm in config)
    features_for_hmm = cfg["model"]["hmm"].get("features_for_hmm")
    if features_for_hmm:
        available = [f for f in features_for_hmm if f in features.columns]
        missing = [f for f in features_for_hmm if f not in features.columns]
        if missing:
            log.warning(f"BIC: configured HMM features not found: {missing}")
        hmm_features = features[available]
        log.info(f"BIC using {len(available)} configured HMM features: {available}")
    else:
        hmm_features = features

    train_features = hmm_features[hmm_features.index < oos_start].dropna()
    log.info(f"BIC training set: {train_features.shape[0]} months x {train_features.shape[1]} features")
    X_train = train_features.values

    selector = HMMSelector(cfg)
    sel = selector.select(X_train)

    rec = stats.bic_recommendation(sel["bic_scores"])
    log.info(f"\nBIC Recommendation:\n  {rec['reasoning']}")
    log.info(f"  BIC scores: {rec['bic_scores']}")

    plots.bic_elbow(sel["bic_scores"], sel["aic_scores"])

    return {
        "best_k": sel["best_k"],
        "bic_scores": sel["bic_scores"],
        "aic_scores": sel["aic_scores"],
        "bic_recommendation": rec,
        "stationarity": stationarity,
        "correlation": corr_matrix,
        "selection_result": sel,
    }


def phase_model(cfg: dict, features, explore_result: dict) -> dict:
    """
    M — Walk-forward HMM training at the BIC-selected K.

    Returns:
        dict with regime_probs, regime_names, full_model

    Caching: saves regime_probs to outputs/models/regime_probs.csv so that
    --phase interpret can be run independently without re-running walk-forward.
    """
    import pickle as _pickle

    log = logging.getLogger("model")
    log.info("=" * 55)
    log.info("  Phase M — MODEL")
    log.info("=" * 55)

    best_k = explore_result["best_k"]
    oos_start = cfg["exploration"]["oos_start"]
    models_dir = Path(cfg["outputs"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_path = models_dir / "regime_probs.csv"
    meta_path = models_dir / "regime_meta.pkl"

    log.info(f"Walk-forward HMM (K={best_k})  OOS from {oos_start} ...")
    trainer = HMMTrainer(cfg)
    # Pass full feature matrix — walk_forward selects the HMM feature subset
    # and handles NaN internally per-window (do NOT dropna here across all 39 cols)
    regime_probs = trainer.walk_forward(features, best_k, oos_start)

    # Fit a full-sample model for interpretation
    full_model = trainer.fit_full(features, best_k, name="hmm_full")

    # Auto-label regimes from emission parameters
    feature_names = (
        trainer._select_features(features).columns.tolist()
        if trainer.features_for_hmm
        else features.columns.tolist()
    )
    regime_names = trainer.label_regimes(full_model, feature_names, best_k)

    # Rename probability columns to human-readable names
    for k, name in enumerate(regime_names):
        old_col = f"prob_{k}"
        new_col = f"prob_{name}"
        if old_col in regime_probs.columns:
            regime_probs.rename(columns={old_col: new_col}, inplace=True)

    log.info(f"Regime names: {regime_names}")
    log.info(f"Regime counts:\n{regime_probs['regime'].value_counts().to_string()}")

    # ── Cache predictions to disk for fast interpret re-runs ─────────────
    regime_probs.to_csv(cache_path)
    with open(meta_path, "wb") as f:
        _pickle.dump({"regime_names": regime_names, "best_k": best_k}, f)
    log.info(f"  [cache] regime_probs saved → {cache_path}")

    return {
        "regime_probs": regime_probs,
        "regime_names": regime_names,
        "full_model": full_model,
        "trainer": trainer,
    }


def load_model_cache(cfg: dict) -> dict | None:
    """
    Load cached regime_probs + full HMM model from disk.
    Returns None if cache files are missing.
    Used by run_pipeline when --phase interpret is run without --phase model.
    """
    import pickle as _pickle

    log = logging.getLogger("model")
    models_dir = Path(cfg["outputs"]["models_dir"])
    cache_path = models_dir / "regime_probs.csv"
    meta_path = models_dir / "regime_meta.pkl"
    model_path = models_dir / "hmm_full.pkl"

    if not all(p.exists() for p in [cache_path, meta_path, model_path]):
        missing = [p for p in [cache_path, meta_path, model_path] if not p.exists()]
        log.warning(f"  [cache] missing model files: {missing}")
        return None

    regime_probs = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    with open(meta_path, "rb") as f:
        meta = _pickle.load(f)
    with open(model_path, "rb") as f:
        full_model = _pickle.load(f)

    trainer = HMMTrainer(cfg)

    log.info(f"  [cache] loaded regime_probs from {cache_path} ({len(regime_probs)} months)")
    log.info(f"  [cache] regime_names: {meta['regime_names']}")

    return {
        "regime_probs": regime_probs,
        "regime_names": meta["regime_names"],
        "full_model": full_model,
        "trainer": trainer,
    }


def phase_interpret(
    cfg: dict,
    model_result: dict,
    asset_returns,
    spy_prices=None,
) -> dict:
    """
    N — Portfolio construction, backtest, metrics, visualisations.

    Returns:
        dict with all_results, numeric_df, display_df, crisis_df
    """
    log = logging.getLogger("interpret")
    log.info("=" * 55)
    log.info("  Phase N — iNTERPRET")
    log.info("=" * 55)

    regime_probs = model_result["regime_probs"]
    regime_names = model_result["regime_names"]
    full_model = model_result["full_model"]

    # Portfolio weights
    allocator = MacroAllocator(cfg)
    prob_cols_regime = [f"prob_{n}" for n in regime_names
                        if f"prob_{n}" in regime_probs.columns]
    # Reconstruct generic prob_N columns for allocator interface
    probs_generic = regime_probs.copy()
    for k, name in enumerate(regime_names):
        if f"prob_{name}" in probs_generic.columns:
            probs_generic[f"prob_{k}"] = probs_generic[f"prob_{name}"]

    target_weights = allocator.compute_weights(probs_generic, regime_names)

    # Optional vol scaling
    target_weights = allocator.apply_vol_scaling(target_weights, asset_returns)

    # Inverse-vol dynamic benchmark weights
    inv_vol_weights = allocator.inverse_vol_static(asset_returns)

    # Simulation
    rebalancer = Rebalancer(cfg)
    sim = rebalancer.simulate(target_weights, asset_returns)
    strategy_returns = sim["portfolio_return"]

    # Save portfolio returns for downstream use (bootstrap, etc.)
    reports_dir = Path(cfg["outputs"]["reports_dir"])
    sim[["portfolio_return", "gross_return", "tc_drag"]].to_csv(
        reports_dir / "portfolio_returns.csv"
    )

    # Backtest vs benchmarks
    backtester = Backtester(cfg)
    all_results = backtester.run(strategy_returns, asset_returns, inv_vol_weights)

    # Metrics
    metrics_engine = PerformanceMetrics(cfg)
    numeric_df, display_df = metrics_engine.comparison_table(all_results)

    log.info(f"\nPerformance Summary:\n{display_df.to_string()}")

    # Crisis analysis
    crisis_df = backtester.crisis_analysis(all_results)
    log.info(f"\nCrisis Period Analysis:\n{crisis_df.to_string(index=False)}")

    # Visualisations
    viz = Visualizer(cfg)

    if spy_prices is not None:
        from src.explore.eda_plots import EDAPlots
        EDAPlots(cfg).regime_timeline(probs_generic, spy_prices, regime_names)

    viz.equity_and_drawdown(all_results)
    viz.allocation_area(target_weights)
    viz.transition_heatmap(full_model.transmat_, regime_names)
    viz.performance_bars(numeric_df)
    viz.rolling_sharpe(all_results)

    log.info("\nAll figures saved to outputs/figures/")

    # ── quantstats HTML tear sheet ────────────────────────────────────────
    reports_dir = Path(cfg["outputs"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    try:
        import quantstats as qs
        spy_bench = all_results.get("Buy & Hold SPY", {})
        bench_ret = spy_bench.get("return") if isinstance(spy_bench, dict) else None
        if bench_ret is None and "Buy & Hold SPY" in all_results:
            bench_ret = all_results["Buy & Hold SPY"]["return"]
        qs.reports.html(
            strategy_returns,
            benchmark=bench_ret,
            output=str(reports_dir / "tearsheet.html"),
            title="Adaptive Macro Regime Strategy — Walk-Forward OOS (2006–2026)",
            periods_per_year=12,
            compounded=True,
        )
        log.info(f"  [qs] tearsheet saved → {reports_dir / 'tearsheet.html'}")
    except Exception as e:
        log.warning(f"  [qs] quantstats tearsheet skipped: {e}")

    return {
        "all_results": all_results,
        "numeric_df": numeric_df,
        "display_df": display_df,
        "crisis_df": crisis_df,
        "target_weights": target_weights,
        "strategy_returns": strategy_returns,
    }


# ── main ─────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    Path("outputs").mkdir(exist_ok=True)
    setup_logging(cfg["project"]["log_level"] if not args.log_level else args.log_level)

    log = logging.getLogger("pipeline")
    log.info(f"\n{'='*60}")
    log.info(f"  {cfg['project']['name']}  v{cfg['project']['version']}")
    log.info(f"{'='*60}")

    phases = args.phase.split(",") if args.phase else ["obtain", "scrub", "explore", "model", "interpret"]

    fred_data, market_data = None, None
    features, df_raw, asset_returns = None, None, None
    explore_result, model_result = None, None

    if "obtain" in phases:
        fred_data, market_data = phase_obtain(cfg, args.refresh)

    if "scrub" in phases:
        if fred_data is None or market_data is None:
            # Load from cache
            fred_data, market_data = phase_obtain(cfg, refresh=False)
        features, df_raw, asset_returns = phase_scrub(cfg, fred_data, market_data)

    if "explore" in phases:
        if features is None:
            if fred_data is None:
                fred_data, market_data = phase_obtain(cfg, refresh=False)
            features, df_raw, asset_returns = phase_scrub(cfg, fred_data, market_data)
        explore_result = phase_explore(cfg, features, df_raw, asset_returns)

    if "model" in phases:
        if explore_result is None:
            raise RuntimeError("Run 'explore' phase before 'model' to determine K via BIC.")
        model_result = phase_model(cfg, features, explore_result)

    if "interpret" in phases:
        # Try to load from cache if model wasn't just trained
        if model_result is None:
            model_result = load_model_cache(cfg)
            if model_result is None:
                raise RuntimeError(
                    "No cached model found. Run --phase scrub,explore,model first."
                )

        # Load asset_returns from cache if scrub wasn't run
        if asset_returns is None:
            ar_path = Path(cfg["outputs"]["processed_dir"]) / "asset_returns.csv"
            if ar_path.exists():
                asset_returns = pd.read_csv(ar_path, index_col=0, parse_dates=True)
                logging.getLogger("pipeline").info(
                    f"  [cache] loaded asset_returns from {ar_path}"
                )
            else:
                raise RuntimeError(
                    "No cached asset_returns found. Run --phase scrub first."
                )

        spy_prices = None
        if market_data and "SPY" in market_data:
            spy_prices = market_data["SPY"]["Close"].resample(
                cfg["data"]["preprocessing"]["resample_freq"]
            ).last()

        phase_interpret(cfg, model_result, asset_returns, spy_prices)

    # ── Robustness grid (optional — runs if 'robustness' in phases) ────
    if "robustness" in phases:
        log.info("\n" + "=" * 55)
        log.info("  Phase R — Robustness Sensitivity Grid")
        log.info("=" * 55)
        grid = RobustnessGrid(cfg)
        rob_df = grid.run()
        log.info(f"  [robustness] {len(rob_df)} grid rows computed.")

    # ── Block bootstrap CI (optional — runs if 'bootstrap' in phases) ──
    if "bootstrap" in phases:
        log.info("\n" + "=" * 55)
        log.info("  Phase B — Block Bootstrap Confidence Intervals")
        log.info("=" * 55)
        bs = BlockBootstrap(cfg)
        boot_results = bs.run()
        log.info(
            f"  [bootstrap] Sharpe {boot_results['sharpe_point']:.3f} "
            f"95% CI [{boot_results['sharpe_ci_lo']:.3f}, {boot_results['sharpe_ci_hi']:.3f}] "
            f"p(>SPY)={boot_results['sharpe_pval_vs_spy']:.3f}"
        )

    # ── Stock overlay (optional — runs if 'overlay' in phases) ─────────
    if "overlay" in phases:
        if market_data is None:
            fred_data, market_data = phase_obtain(cfg, refresh=False)
        if model_result is None:
            model_result = load_model_cache(cfg)
        if model_result is not None and market_data is not None:
            _run_stock_overlay(cfg, market_data, model_result["regime_probs"])
        else:
            log.warning("Stock overlay skipped: need market data and model cache.")

    log.info("\nOK Pipeline complete.  Results in outputs/")


_OVERLAY_KEY_EVENTS: dict[str, list[dict]] = {
    "NVDA": [
        {"date": "2022-01-31", "label": "2022 Bear Start", "color": "#e74c3c"},
        {"date": "2022-10-31", "label": "Bear Bottom", "color": "#c0392b"},
        {"date": "2023-03-31", "label": "ChatGPT Boom", "color": "#27ae60"},
        {"date": "2024-06-28", "label": "3:10 Split", "color": "#3498db"},
    ],
    "WDC": [
        {"date": "2015-10-31", "label": "SanDisk Acquisition", "color": "#e67e22"},
        {"date": "2020-03-31", "label": "COVID Crash", "color": "#e74c3c"},
        {"date": "2025-02-28", "label": "SNDK Spinoff", "color": "#9b59b6"},
    ],
}


def _run_stock_overlay(
    cfg: dict, market_data: dict, regime_probs: pd.DataFrame
) -> None:
    """Run the CompositeSignal stock overlay for configured tickers."""
    log = logging.getLogger("overlay")
    log.info("=" * 55)
    log.info("  Stock Overlay — Composite Signal")
    log.info("=" * 55)

    cs = CompositeSignal(cfg)
    viz = Visualizer(cfg)
    tickers = cfg.get("stock_overlay", {}).get("tickers", ["NVDA", "WDC"])
    out_dir = Path(cfg["outputs"]["processed_dir"])

    spy_daily = None
    if "SPY" in market_data:
        spy_daily = market_data["SPY"]["Close"]

    for ticker in tickers:
        if ticker not in market_data:
            log.warning(f"  [{ticker}] not in market_data — skipping")
            continue
        daily_prices = market_data[ticker]
        if spy_daily is None:
            log.warning(f"  [{ticker}] SPY data missing — correlation signal zeroed")
            spy_daily = pd.Series(0.0, index=daily_prices.index)

        try:
            signal_df = cs.compute(ticker, daily_prices, spy_daily, regime_probs)
            out_path = out_dir / f"overlay_{ticker.lower()}.csv"
            signal_df.to_csv(out_path)
            log.info(f"  [{ticker}] signals saved → {out_path.name}")

            key_events = _OVERLAY_KEY_EVENTS.get(ticker)
            viz.overlay_signal(ticker, signal_df, key_events)
        except Exception as e:
            log.error(f"  [{ticker}] overlay failed: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Macro Regime Detection & Factor Rotation Strategy — OSEMN Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force re-download of all data (bypass cache)",
    )
    p.add_argument(
        "--phase",
        default=None,
        help="Comma-separated phases to run: obtain,scrub,explore,model,interpret,"
             "overlay,robustness,bootstrap. Default: all phases.",
    )
    p.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity override",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
