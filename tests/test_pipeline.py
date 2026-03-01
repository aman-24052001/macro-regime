"""
tests/test_pipeline.py — Core pipeline regression tests

Run with:  .venv/Scripts/python.exe -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    return load_config("config/config.yaml")


@pytest.fixture(scope="module")
def sample_daily_prices():
    """200 days of synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    return pd.DataFrame(
        {"Close": close, "Volume": rng.integers(1_000_000, 10_000_000, n)},
        index=dates,
    )


@pytest.fixture(scope="module")
def sample_monthly_regime_probs():
    """24 months of synthetic regime probability data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-31", periods=24, freq="BME")
    probs = rng.dirichlet([2, 3, 1], size=24)   # stagnation-heavy
    return pd.DataFrame(
        probs, index=dates, columns=["prob_expansion", "prob_stagnation", "prob_contraction"]
    )


# ── Config loader tests ───────────────────────────────────────────────────

class TestConfigLoader:

    def test_loads_without_error(self, cfg):
        assert cfg is not None

    def test_required_top_level_keys(self, cfg):
        for key in ["data", "feature_engineering", "model", "portfolio", "backtest", "outputs"]:
            assert key in cfg, f"Missing top-level key: {key}"

    def test_etfs_defined(self, cfg):
        etfs = cfg["data"]["market"]["etfs"]
        assert set(etfs.keys()) == {"SPY", "TLT", "GLD", "LQD", "HYG", "BIL"}

    def test_hmm_features_list(self, cfg):
        feats = cfg["model"]["hmm"]["features_for_hmm"]
        assert len(feats) == 7   # 7 curated HMM features

    def test_force_k_is_3(self, cfg):
        assert cfg["model"]["hmm"]["force_k"] == 3

    def test_walk_forward_n_fits_restored(self, cfg):
        """Ensure n_fits was restored to 10 (not left at 1 from cache generation)."""
        assert cfg["model"]["hmm"]["walk_forward_n_fits"] == 10

    def test_contraction_portfolio_sums_to_1(self, cfg):
        cont = cfg["portfolio"]["regime_portfolios"]["contraction"]
        total = sum(cont.values())
        assert abs(total - 1.0) < 1e-9, f"Contraction weights sum to {total}"

    def test_stock_overlay_config_present(self, cfg):
        assert "stock_overlay" in cfg
        assert cfg["stock_overlay"]["max_weight"] == 0.25


# ── Feature engineering tests ─────────────────────────────────────────────

class TestFeatureEngineer:

    def test_34_columns(self):
        """Post-cleanup: feature matrix must have exactly 34 columns."""
        from src.scrub.feature_engineer import FeatureEngineer
        from src.config_loader import load_config
        import warnings
        cfg = load_config("config/config.yaml")

        # Build minimal raw DataFrame with required columns
        rng = np.random.default_rng(0)
        n = 60
        dates = pd.date_range("2015-01-31", periods=n, freq="BME")
        cols = [
            "yield_curve_10y2y", "yield_curve_10y3m", "credit_spread_baa",
            "gdp_real", "cpi", "unemployment", "fed_funds",
            "industrial_production", "m2_money", "jobless_claims",
        ]
        raw = pd.DataFrame(rng.standard_normal((n, len(cols))), index=dates, columns=cols)
        raw_pos = raw.abs() + 0.01   # ensure positive values for pct_change

        # Z-scored input (same shape, we just need non-NaN)
        z = raw.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fe = FeatureEngineer(cfg)
            full = fe.engineer(raw_pos, z, save=False)

        assert full.shape[1] == 34, (
            f"Expected 34 feature columns after cleanup, got {full.shape[1]}: {list(full.columns)}"
        )

    def test_no_exact_duplicate_columns(self):
        """Correlation = 1.00 pairs must not exist in output."""
        processed = Path("data/processed/features_full.csv")
        if not processed.exists():
            pytest.skip("features_full.csv not generated yet")
        df = pd.read_csv(processed, index_col=0, parse_dates=True).dropna()
        corr = df.corr().abs()
        # Find off-diagonal pairs with corr == 1.0
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = [(c, r) for c in upper.columns for r in upper.index if upper.loc[r, c] == 1.0]
        assert len(pairs) == 0, f"Exact duplicate feature pairs found: {pairs}"


# ── Rebalancer date index tests ───────────────────────────────────────────

class TestRebalancer:

    def test_output_index_matches_oos_window(self, cfg):
        """
        Regression test for Bug 6: rebalancer output must be indexed by
        the actual OOS dates (2006+), not the first N dates of all returns.
        """
        from src.portfolio.rebalancer import Rebalancer

        rng = np.random.default_rng(0)
        # Simulate a 120-month return series starting 2000 (24 pre-OOS + 96 OOS)
        all_dates = pd.date_range("2000-01-31", periods=120, freq="BME")
        assets = ["SPY", "TLT", "GLD", "LQD", "HYG", "BIL"]
        asset_returns = pd.DataFrame(
            rng.normal(0.005, 0.04, (120, len(assets))),
            index=all_dates,
            columns=assets,
        )

        # Weights start at 2004-01-31 (simulate OOS starting late)
        w_dates = pd.date_range("2004-01-31", periods=96, freq="BME")
        weights = pd.DataFrame(
            np.tile([0.1667] * 6, (96, 1)),
            index=w_dates,
            columns=assets,
        )

        reb = Rebalancer(cfg)
        result = reb.simulate(weights, asset_returns)

        # All result dates must be >= the first weight date (no pre-weight dates)
        assert not result.empty, "Rebalancer returned empty DataFrame"
        first_weight_date = weights.index[0]
        assert result.index[0] >= first_weight_date, (
            f"Bug 6 regression: result starts at {result.index[0]} "
            f"which is before first weight date {first_weight_date}"
        )

    def test_result_length_matches_date_count(self, cfg):
        """Each row must correspond to exactly one date."""
        from src.portfolio.rebalancer import Rebalancer

        rng = np.random.default_rng(1)
        dates = pd.date_range("2006-01-31", periods=100, freq="BME")
        assets = ["SPY", "TLT", "GLD", "LQD", "HYG", "BIL"]
        asset_returns = pd.DataFrame(
            rng.normal(0.005, 0.03, (100, 6)), index=dates, columns=assets
        )
        weights = pd.DataFrame(
            np.tile([1/6] * 6, (100, 1)), index=dates, columns=assets
        )

        reb = Rebalancer(cfg)
        result = reb.simulate(weights, asset_returns)

        # Index should be unique dates
        assert result.index.is_unique, "Result index has duplicate dates"
        assert len(result) <= len(dates), "Result has more rows than input dates"


# ── Composite signal tests ─────────────────────────────────────────────────

class TestCompositeSignal:

    def test_output_columns(self, cfg, sample_daily_prices, sample_monthly_regime_probs):
        from src.model.composite_signal import CompositeSignal
        cs = CompositeSignal(cfg)
        spy = sample_daily_prices["Close"]
        result = cs.compute("TEST", sample_daily_prices, spy, sample_monthly_regime_probs)

        expected_cols = {
            "sig_momentum", "sig_volatility", "sig_trend", "sig_correlation",
            "sig_macro", "composite_score", "sigma_annual", "macro_scale", "weight",
        }
        assert expected_cols == set(result.columns), f"Missing: {expected_cols - set(result.columns)}"

    def test_weight_bounded(self, cfg, sample_daily_prices, sample_monthly_regime_probs):
        from src.model.composite_signal import CompositeSignal
        cs = CompositeSignal(cfg)
        spy = sample_daily_prices["Close"]
        result = cs.compute("TEST", sample_daily_prices, spy, sample_monthly_regime_probs)

        assert (result["weight"] >= 0).all(), "Negative weights found"
        assert (result["weight"] <= cs.w_max + 1e-9).all(), "Weight exceeds max"

    def test_signal_weights_sum_to_one(self, cfg):
        from src.model.composite_signal import CompositeSignal
        cs = CompositeSignal(cfg)
        total = sum(cs.signal_weights.values())
        assert abs(total - 1.0) < 1e-9, f"Signal weights sum to {total}"

    def test_signals_in_minus1_to_plus1(self, cfg, sample_daily_prices, sample_monthly_regime_probs):
        from src.model.composite_signal import CompositeSignal
        cs = CompositeSignal(cfg)
        spy = sample_daily_prices["Close"]
        result = cs.compute("TEST", sample_daily_prices, spy, sample_monthly_regime_probs)

        for col in ["sig_momentum", "sig_volatility", "sig_trend", "sig_macro"]:
            valid = result[col].dropna()
            assert (valid >= -1.01).all() and (valid <= 1.01).all(), (
                f"{col} out of [-1, 1] range: min={valid.min():.3f}, max={valid.max():.3f}"
            )


# ── Rolling Z-score tests ─────────────────────────────────────────────────

class TestRollingZscore:

    def test_zscore_near_zero_mean(self):
        """Rolling Z-score of a stationary series should have approximately zero mean."""
        from src.scrub.feature_engineer import FeatureEngineer
        from src.config_loader import load_config

        cfg = load_config("config/config.yaml")
        fe = FeatureEngineer(cfg)

        rng = np.random.default_rng(0)
        n = 200
        s = pd.Series(rng.standard_normal(n) + 0.1)
        z = fe._rolling_zscore(s).dropna()

        # After normalization, mean should be near 0 and std near 1
        assert abs(z.mean()) < 0.5, f"Z-score mean {z.mean():.3f} too large"
        assert 0.5 < z.std() < 1.5, f"Z-score std {z.std():.3f} unexpected"
