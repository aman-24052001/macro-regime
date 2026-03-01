from __future__ import annotations
"""
scrub/feature_engineer.py  —  S in OSEMN
------------------------------------------
Builds derived macro features on top of the raw monthly series.

All feature names, windows, and weights come from config.
The output is a complete, Z-scored feature matrix ready for EDA and HMM.

Derived features
────────────────
  • YoY growth rates        (GDP, CPI, IP, M2)
  • Momentum / rate-of-change at multiple windows
  • Yield curve: level, slope, inversion binary flag
  • Credit impulse          (Δ credit spread)
  • Unemployment delta      (3M and 12M)
  • Financial Stress Index  (weighted composite, configurable components)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute derived features from raw monthly FRED data."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        fe = cfg["feature_engineering"]

        self.momentum_windows: list[int] = fe["momentum_windows"]
        self.yoy_features: list[str] = fe["yoy_features"]
        self.stress_cfg: dict = fe["stress_index"]

        prep = cfg["data"]["preprocessing"]
        self.zscore_window: int = prep["zscore_window"]
        self.zscore_min_periods: int = prep["zscore_min_periods"]

        self.processed_dir = Path(cfg["outputs"]["processed_dir"])

    # ── public ───────────────────────────────────────────────────────────

    def engineer(
        self,
        df_raw: pd.DataFrame,
        df_zscored: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Build derived features and return the full feature matrix.

        Args:
            df_raw     : raw monthly levels (from Preprocessor)
            df_zscored : rolling Z-scored base features

        Returns:
            Full feature matrix (base Z-scored + derived Z-scored),
            ready for EDA and HMM training.
        """
        derived = pd.DataFrame(index=df_raw.index)

        derived = self._add_yield_curve_features(derived, df_raw)
        derived = self._add_yoy_growth(derived, df_raw)
        derived = self._add_momentum_features(derived, df_raw)
        derived = self._add_unemployment_delta(derived, df_raw)
        derived = self._add_stress_index(derived, df_zscored)

        # Z-score the derived features with the same rolling window
        derived_z = derived.apply(self._rolling_zscore)

        # Merge with base Z-scored features
        full = pd.concat([df_zscored, derived_z], axis=1)
        # Drop any exact duplicate column names that may have crept in
        full = full.loc[:, ~full.columns.duplicated()]

        logger.info(
            f"Feature matrix after engineering: "
            f"{full.shape[0]} rows × {full.shape[1]} columns"
        )
        logger.info(f"  Features: {list(full.columns)}")

        if save:
            full.to_csv(self.processed_dir / "features_full.csv")

        return full

    # ── private feature builders ─────────────────────────────────────────

    def _add_yield_curve_features(
        self, derived: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        if "yield_curve_10y2y" in raw.columns:
            # Inversion flag: 1 when curve is inverted (recession signal)
            derived["yc_inverted"] = (raw["yield_curve_10y2y"] < 0).astype(float)
            # Note: yc_slope_10y2y omitted — identical to yield_curve_10y2y (corr=1.00)

        if "yield_curve_10y3m" in raw.columns:
            derived["yc_10y3m_inverted"] = (raw["yield_curve_10y3m"] < 0).astype(float)
            # Note: yc_slope_10y3m omitted — identical to yield_curve_10y3m (corr=1.00)

        return derived

    def _add_yoy_growth(
        self, derived: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        """12-month percent change and 3-month acceleration for configured series."""
        for series_name in self.yoy_features:
            if series_name not in raw.columns:
                continue
            yoy_col = f"{series_name}_yoy"
            derived[yoy_col] = raw[series_name].pct_change(12, fill_method=None) * 100
            # Acceleration = change in YoY growth rate over 3 months
            derived[f"{series_name}_accel"] = derived[yoy_col].diff(3)
        return derived

    def _add_momentum_features(
        self, derived: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        """Rate-of-change at multiple windows for key series."""
        target_cols = [
            "yield_curve_10y2y",
            "credit_spread_baa",
            "unemployment",
            "fed_funds",
        ]
        for col in target_cols:
            if col not in raw.columns:
                continue
            for w in self.momentum_windows:
                derived[f"{col}_mom{w}m"] = raw[col].diff(w)
        return derived

    def _add_unemployment_delta(
        self, derived: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        col = "unemployment"
        if col not in raw.columns:
            return derived
        # unemp_delta_3m omitted — identical to unemployment_mom3m (corr=1.00)
        derived["unemp_delta_12m"] = raw[col].diff(12)
        return derived

    def _add_stress_index(
        self, derived: pd.DataFrame, df_zscored: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Financial Stress Index = weighted sum of Z-scored component series.
        Components and weights defined in config['feature_engineering']['stress_index'].
        Yield curve component is sign-flipped (inversion = stress = positive FSI).
        """
        components = self.stress_cfg["components"]
        weights = self.stress_cfg["weights"]
        name = self.stress_cfg.get("name", "financial_stress_index")

        available = [c for c in components if c in df_zscored.columns]
        if not available:
            logger.warning("  Stress index: no components found in Z-scored features.")
            return derived

        stress = pd.DataFrame(index=df_zscored.index)
        total_w = 0.0

        for comp, w in zip(components, weights):
            if comp not in df_zscored.columns:
                continue
            col_data = df_zscored[comp].copy()
            # Yield curve: inverted = stress → flip sign
            if "yield_curve" in comp:
                col_data = -col_data
            stress[comp] = col_data * w
            total_w += w

        derived[name] = stress.sum(axis=1) / total_w if total_w > 0 else np.nan
        return derived

    # ── helper ───────────────────────────────────────────────────────────

    def _rolling_zscore(self, s: pd.Series) -> pd.Series:
        mu = s.rolling(window=self.zscore_window, min_periods=self.zscore_min_periods).mean()
        sigma = s.rolling(
            window=self.zscore_window, min_periods=self.zscore_min_periods
        ).std()
        return (s - mu) / sigma.replace(0.0, np.nan)
