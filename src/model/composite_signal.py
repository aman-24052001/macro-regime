from __future__ import annotations
"""
model/composite_signal.py  —  Stock Overlay Layer
---------------------------------------------------
Multi-signal adaptive model for individual stock position sizing.
Implements the framework from adaptive_model.md:

  W(stock, t) = composite_signal(t) × macro_modifier(t) / (1 + λ × σ_forecast(t))

Signal components (weights from spec):
  π₁ = 0.25  Momentum   : SMA-50/200 crossover + RSI normalised
  π₂ = 0.20  Volatility : GARCH(1,1) conditional vol regime
  π₃ = 0.30  Trend      : 12M price momentum (fundamental growth proxy)
  π₄ = 0.15  Correlation: Rolling corr vs SPY (decoupling = idiosyncratic)
  π₅ = 0.10  Macro      : P(expansion) from macro HMM

Position size:
  raw_score   = Σ πᵢ × Normalised_Signal_i(t)          ∈ [-1, +1]
  macro_scale = P(expansion) × 1.5 + P(stagnation) × 1.0 + P(contraction) × 0.5
  vol_penalty = 1 / (1 + λ × σ_forecast / σ_target)
  weight      = clip(raw_score × macro_scale × vol_penalty, 0, w_max)

Case studies supported: NVDA (AI transformation), WDC/SNDK (M&A event-driven)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default signal weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "momentum":    0.25,
    "volatility":  0.20,
    "trend":       0.30,
    "correlation": 0.15,
    "macro":       0.10,
}

# Macro regime multipliers (Expansion / Stagnation / Contraction)
MACRO_MULT = {"expansion": 1.50, "stagnation": 1.00, "contraction": 0.50}


class CompositeSignal:
    """
    Compute regime-adaptive composite signal and position size for a stock.

    Usage:
        cs = CompositeSignal(cfg)
        signal_df = cs.compute(ticker, daily_prices, spy_prices, regime_probs)
        # Returns DataFrame with individual signals, composite score, and weight
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        overlay_cfg = cfg.get("stock_overlay", {})

        self.w_max: float   = overlay_cfg.get("max_weight",     0.25)
        self.lambda_: float = overlay_cfg.get("lambda_risk",    0.30)
        self.vol_target: float = overlay_cfg.get("vol_target",  0.30)  # 30% ann. vol target
        self.corr_window: int  = overlay_cfg.get("corr_window", 60)    # trading days
        self.trend_window: int = overlay_cfg.get("trend_window", 252)  # days (12M)
        self.signal_weights: dict = overlay_cfg.get("signal_weights", DEFAULT_WEIGHTS)

        # Normalise weights in case they don't sum to 1
        total = sum(self.signal_weights.values())
        self.signal_weights = {k: v / total for k, v in self.signal_weights.items()}

    # ── public ───────────────────────────────────────────────────────────

    def compute(
        self,
        ticker: str,
        daily_prices: pd.DataFrame,
        spy_prices: pd.Series,
        regime_probs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute composite signal and recommended weight at monthly frequency.

        Args:
            ticker       : e.g. 'NVDA' or 'WDC'
            daily_prices : OHLCV DataFrame with columns Close, Volume (daily)
            spy_prices   : SPY Close series (daily), for correlation signal
            regime_probs : monthly DataFrame with columns prob_expansion,
                           prob_stagnation, prob_contraction (from HMM)

        Returns:
            Monthly DataFrame with columns:
              sig_momentum, sig_volatility, sig_trend, sig_correlation,
              sig_macro, composite_score, sigma_annual, macro_scale, weight
        """
        close = daily_prices["Close"].dropna()
        returns_daily = close.pct_change().dropna()

        # ── Individual signals (daily) ─────────────────────────────────
        sig_mom   = self._momentum_signal(close)
        sig_vol   = self._volatility_signal(returns_daily)
        sig_trend = self._trend_signal(close)
        sig_corr  = self._correlation_signal(returns_daily, spy_prices)

        # ── Resample to monthly (last business day) ────────────────────
        freq = "BME"
        monthly_close = close.resample(freq).last()

        def _monthly_last(s: pd.Series) -> pd.Series:
            return s.resample(freq).last()

        mom_m   = _monthly_last(sig_mom)
        vol_m   = _monthly_last(sig_vol["signal"])
        vol_raw = _monthly_last(sig_vol["vol_annual"])   # raw annualised vol
        trend_m = _monthly_last(sig_trend)
        corr_m  = _monthly_last(sig_corr)

        # ── Macro signal from HMM probabilities ───────────────────────
        macro_m = self._macro_signal(regime_probs)

        # ── Align all signals on common monthly index ──────────────────
        base_idx = monthly_close.index
        mom_m   = mom_m.reindex(base_idx)
        vol_m   = vol_m.reindex(base_idx)
        vol_raw = vol_raw.reindex(base_idx)
        trend_m = trend_m.reindex(base_idx)
        corr_m  = corr_m.reindex(base_idx)
        macro_m = macro_m.reindex(base_idx)

        # ── Composite score ────────────────────────────────────────────
        w = self.signal_weights
        composite = (
            w["momentum"]    * mom_m.fillna(0)
            + w["volatility"]  * vol_m.fillna(0)
            + w["trend"]       * trend_m.fillna(0)
            + w["correlation"] * corr_m.fillna(0)
            + w["macro"]       * macro_m.fillna(0)
        )

        # ── Macro regime scale ─────────────────────────────────────────
        macro_scale = self._macro_scale(regime_probs, base_idx)

        # ── Volatility penalty ─────────────────────────────────────────
        sigma = vol_raw.fillna(vol_raw.median())
        vol_penalty = 1.0 / (1.0 + self.lambda_ * sigma / self.vol_target)

        # ── Final weight ───────────────────────────────────────────────
        raw_weight = composite.clip(lower=0) * macro_scale * vol_penalty
        weight = raw_weight.clip(upper=self.w_max)

        df = pd.DataFrame(
            {
                "sig_momentum":   mom_m,
                "sig_volatility": vol_m,
                "sig_trend":      trend_m,
                "sig_correlation": corr_m,
                "sig_macro":      macro_m,
                "composite_score": composite,
                "sigma_annual":    sigma,
                "macro_scale":     macro_scale,
                "weight":          weight,
            },
            index=base_idx,
        )
        df.index.name = "date"

        # Log summary
        logger.info(
            f"[{ticker}] composite signal: "
            f"{len(df)} months  "
            f"avg weight={weight.mean():.1%}  "
            f"max weight={weight.max():.1%}"
        )
        return df

    # ── Signal builders ───────────────────────────────────────────────

    def _momentum_signal(self, close: pd.Series) -> pd.Series:
        """
        SMA-50/200 crossover + RSI(14) normalised.
        Output ∈ [-1, +1].
        """
        sma50  = close.rolling(50,  min_periods=30).mean()
        sma200 = close.rolling(200, min_periods=120).mean()

        # SMA crossover: normalised spread
        sigma_price = close.rolling(200, min_periods=60).std()
        ma_cross = (sma50 - sma200) / sigma_price.replace(0, np.nan)
        ma_cross_norm = np.tanh(ma_cross)   # squash to [-1, +1]

        # RSI(14) normalised to [-1, +1]
        rsi = _compute_rsi(close, period=14)
        rsi_norm = (rsi - 50) / 50          # 0 RSI → -1, 100 → +1

        signal = 0.60 * ma_cross_norm + 0.40 * rsi_norm
        return signal.rename("sig_momentum")

    def _volatility_signal(self, returns_daily: pd.Series) -> pd.DataFrame:
        """
        Rolling realised vol regime.
        Signal ∈ [-1, +1]:  -1 = crisis vol (reduce), +1 = normal/low vol (expand)

        Returns DataFrame with 'signal' and 'vol_annual' columns.
        """
        # 60-day rolling annualised volatility
        vol = returns_daily.rolling(60, min_periods=30).std() * np.sqrt(252)

        # Rolling vol-of-vol normalisation
        vol_mean  = vol.rolling(252, min_periods=120).mean()
        vol_std   = vol.rolling(252, min_periods=120).std()
        vol_z     = (vol - vol_mean) / vol_std.replace(0, np.nan)

        # Invert: high vol → negative signal (reduce position)
        signal = -np.tanh(vol_z)    # high vol z = large negative signal

        return pd.DataFrame({"signal": signal, "vol_annual": vol})

    def _trend_signal(self, close: pd.Series) -> pd.Series:
        """
        12-month price momentum (trend / fundamental proxy).
        Normalised to [-1, +1] using sign + log magnitude.
        """
        mom12 = close.pct_change(self.trend_window)   # 252 trading days ≈ 12 months
        # Normalise with rolling cross-sectional z-score
        mu = mom12.rolling(252, min_periods=120).mean()
        sd = mom12.rolling(252, min_periods=120).std()
        z  = (mom12 - mu) / sd.replace(0, np.nan)
        return np.tanh(z / 1.5).rename("sig_trend")

    def _correlation_signal(
        self, returns_daily: pd.Series, spy_prices: pd.Series
    ) -> pd.Series:
        """
        Rolling correlation with SPY.
        Interpretation:
          - High corr  → beta/systematic → neutral (0)
          - Low corr   → idiosyncratic driver → signal based on stock direction
          - Decoupling WITH upward momentum → strongest positive signal

        Output ∈ [-1, +1].
        """
        spy_ret = spy_prices.pct_change().dropna()
        aligned = pd.concat([returns_daily, spy_ret], axis=1, join="inner").dropna()
        if aligned.shape[1] < 2:
            return pd.Series(0.0, index=returns_daily.index, name="sig_correlation")

        aligned.columns = ["stock", "spy"]
        roll_corr = aligned["stock"].rolling(self.corr_window, min_periods=40).corr(aligned["spy"])

        # Low correlation = idiosyncratic driver
        # If also in uptrend (stock > spy return over period): positive signal
        # If low corr but stock declining: negative
        stock_drift = aligned["stock"].rolling(self.corr_window, min_periods=40).mean()
        direction   = np.sign(stock_drift).fillna(0)

        idio_signal = (1.0 - roll_corr.abs()) * direction  # [-1, +1]
        return idio_signal.rename("sig_correlation")

    def _macro_signal(self, regime_probs: pd.DataFrame) -> pd.Series:
        """
        Macro regime signal from HMM probabilities.
        P(expansion) → +1, P(contraction) → -1, P(stagnation) → 0.
        """
        exp_col  = _find_col(regime_probs, ["prob_expansion",  "prob_0"])
        stag_col = _find_col(regime_probs, ["prob_stagnation", "prob_1"])
        cont_col = _find_col(regime_probs, ["prob_contraction","prob_2"])

        if exp_col is None:
            logger.warning("Macro signal: could not find regime probability columns")
            return pd.Series(0.0, index=regime_probs.index, name="sig_macro")

        p_exp  = regime_probs.get(exp_col,  pd.Series(0.0, index=regime_probs.index))
        p_stag = regime_probs.get(stag_col, pd.Series(0.0, index=regime_probs.index))
        p_cont = regime_probs.get(cont_col, pd.Series(0.0, index=regime_probs.index))

        # Weighted: expansion = +1, stagnation = 0, contraction = -1
        signal = p_exp * 1.0 + p_stag * 0.0 + p_cont * (-1.0)
        return signal.rename("sig_macro")

    def _macro_scale(
        self, regime_probs: pd.DataFrame, index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Macro scale multiplier: Expansion = 1.5×, Stagnation = 1.0×, Contraction = 0.5×.
        """
        exp_col  = _find_col(regime_probs, ["prob_expansion",  "prob_0"])
        stag_col = _find_col(regime_probs, ["prob_stagnation", "prob_1"])
        cont_col = _find_col(regime_probs, ["prob_contraction","prob_2"])

        if exp_col is None:
            return pd.Series(1.0, index=index)

        p_exp  = regime_probs.get(exp_col,  pd.Series(0.0, index=regime_probs.index))
        p_stag = regime_probs.get(stag_col, pd.Series(0.0, index=regime_probs.index))
        p_cont = regime_probs.get(cont_col, pd.Series(0.0, index=regime_probs.index))

        scale = (
            p_exp  * MACRO_MULT["expansion"]
            + p_stag * MACRO_MULT["stagnation"]
            + p_cont * MACRO_MULT["contraction"]
        ).reindex(index, method="ffill")

        return scale.fillna(1.0)


# ── module-level helpers ──────────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
