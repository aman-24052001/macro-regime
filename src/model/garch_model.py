from __future__ import annotations
"""
model/garch_model.py  —  M in OSEMN
--------------------------------------
GARCH(p,q) conditional volatility estimation for the stock overlay layer.

Used to:
  1. Estimate current conditional volatility σ(t) — feeds into the
     adaptive position-sizing formula as σ_forecast
  2. Regime interpretation — high α (news-sensitivity) vs high β (persistence)

All model orders and distributions come from config.
"""

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GARCHModel:
    """Fit GARCH models for conditional volatility estimation."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        garch_cfg = cfg["model"]["garch"]

        self.p: int = garch_cfg["p"]
        self.q: int = garch_cfg["q"]
        self.vol_model: str = garch_cfg["vol_model"]    # 'GARCH' | 'EGARCH' | 'GJR-GARCH'
        self.dist: str = garch_cfg["dist"]              # 'normal' | 't' | 'skewt'
        self.rolling_window: int = garch_cfg["rolling_window"]  # months

    # ── public ───────────────────────────────────────────────────────────

    def fit(self, returns: pd.Series) -> dict:
        """
        Fit GARCH on a return series.

        Args:
            returns : monthly return series (fraction, e.g. 0.05 = 5%)

        Returns dict with:
            conditional_vol  — annualised conditional volatility series
            params           — {omega, alpha, beta}
            persistence      — alpha + beta  (close to 1 = long memory)
            half_life        — months for vol shock to halve
            aic, bic         — model information criteria
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("Install 'arch' package:  pip install arch")

        r_pct = returns.dropna() * 100   # arch works in percentage units

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                r_pct,
                vol=self.vol_model,
                p=self.p,
                q=self.q,
                dist=self.dist,
            )
            result = model.fit(disp="off", show_warning=False)

        # Rescale conditional vol: % → annual fraction
        cond_vol = result.conditional_volatility / 100 * np.sqrt(12)

        params = result.params
        alpha = float(params.get("alpha[1]", params.get("alpha", np.nan)))
        beta = float(params.get("beta[1]", params.get("beta", np.nan)))
        omega = float(params.get("omega", np.nan))
        persistence = alpha + beta if not (np.isnan(alpha) or np.isnan(beta)) else np.nan

        # Volatility half-life (months): how quickly a shock decays
        half_life = (
            np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.nan
        )

        # Long-run (unconditional) annualised volatility
        long_run_vol = (
            np.sqrt(omega / (1 - persistence)) / 100 * np.sqrt(12)
            if 0 < persistence < 1
            else np.nan
        )

        logger.info(
            f"GARCH({self.p},{self.q})  ω={omega:.5f}  "
            f"α={alpha:.4f}  β={beta:.4f}  "
            f"persistence={persistence:.4f}  "
            f"half-life={half_life:.1f}m  "
            f"LR-vol={long_run_vol:.1%}"
        )

        return {
            "conditional_vol": cond_vol,
            "params": {"omega": omega, "alpha": alpha, "beta": beta},
            "persistence": persistence,
            "half_life_months": half_life,
            "long_run_vol_annual": long_run_vol,
            "aic": result.aic,
            "bic": result.bic,
            "fit_result": result,
        }

    def rolling_forecast(self, returns: pd.Series) -> pd.Series:
        """
        Expanding-window 1-step-ahead volatility forecast.
        For each date T, train on returns[:T] and forecast σ at T+1.

        Returns:
            Series of 1-step-ahead annualised conditional volatility.
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("Install 'arch' package:  pip install arch")

        r_clean = returns.dropna()
        forecasts: dict[pd.Timestamp, float] = {}
        dates = r_clean.index

        for i in range(self.rolling_window, len(dates)):
            window = r_clean.iloc[i - self.rolling_window : i]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = arch_model(
                        window * 100, vol=self.vol_model, p=self.p, q=self.q, dist=self.dist
                    )
                    fit = m.fit(disp="off", show_warning=False)
                    fc = fit.forecast(horizon=1, reindex=False)
                    vol_1step = float(np.sqrt(fc.variance.values[-1, 0])) / 100 * np.sqrt(12)
                    forecasts[dates[i]] = vol_1step
            except Exception as exc:
                logger.debug(f"  GARCH rolling {dates[i].date()}: {exc}")

        return pd.Series(forecasts, name="garch_vol_1step_annual")

    def compare_specifications(self, returns: pd.Series) -> pd.DataFrame:
        """
        Fit multiple GARCH specifications and compare by BIC.
        Useful in the model notebook to choose vol_model and dist from data.
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("Install 'arch' package:  pip install arch")

        specs = [
            ("GARCH", "normal"),
            ("GARCH", "t"),
            ("EGARCH", "normal"),
            ("EGARCH", "t"),
            ("GARCH", "skewt"),
        ]
        rows = []
        r_pct = returns.dropna() * 100

        for vol, dist in specs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = arch_model(r_pct, vol=vol, p=self.p, q=self.q, dist=dist)
                    fit = m.fit(disp="off", show_warning=False)
                    rows.append(
                        {
                            "vol_model": vol,
                            "distribution": dist,
                            "aic": round(fit.aic, 2),
                            "bic": round(fit.bic, 2),
                            "log_likelihood": round(fit.loglikelihood, 2),
                        }
                    )
            except Exception as exc:
                logger.debug(f"  spec ({vol},{dist}) failed: {exc}")

        return pd.DataFrame(rows).sort_values("bic")
