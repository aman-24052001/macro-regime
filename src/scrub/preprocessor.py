from __future__ import annotations
"""
scrub/preprocessor.py  —  S in OSEMN
---------------------------------------
Converts raw FRED series (mixed daily/weekly/monthly/quarterly) and
raw market prices into two clean artefacts:

  1. feature_matrix  — Z-scored monthly macro features ready for HMM
  2. returns_matrix  — monthly log-returns for portfolio assets

All parameters (resample freq, Z-score window, fill limits) come
from config.  No magic numbers here.

Alignment contract
──────────────────
  Daily   → month-end last observation
  Weekly  → month-end last observation
  Monthly → align to Business-Month-End (BME)
  Quarterly → linear interpolation to monthly, forward-filled ≤ max gap
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Preprocessor:
    """Transform raw data → clean, normalised monthly feature matrix."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        prep = cfg["data"]["preprocessing"]

        self.resample_freq: str = prep["resample_freq"]           # "BME"
        self.zscore_window: int = prep["zscore_window"]           # 36
        self.zscore_min_periods: int = prep["zscore_min_periods"] # 12
        self.max_fill: int = prep["max_fwd_fill_months"]          # 3
        self.interp_method: str = prep["quarterly_interp_method"] # "linear"
        self.missing_threshold: float = prep["missing_col_threshold"] # 0.10

        self.series_meta: dict = cfg["data"]["fred"]["series"]
        self.processed_dir = Path(cfg["outputs"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── public ───────────────────────────────────────────────────────────

    def build_feature_matrix(
        self,
        fred_data: dict[str, pd.Series],
        save: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align FRED series → monthly, merge, normalise.

        Returns:
            df_zscored : rolling Z-scored feature matrix  (HMM input)
            df_raw     : raw monthly levels               (feature engineering input)
        """
        monthly: dict[str, pd.Series] = {}
        for name, series in fred_data.items():
            logger.debug(f"  Resampling {name}")
            monthly[name] = self._resample(series, name)

        df_raw = pd.DataFrame(monthly)

        # Extend credit_spread_baa back to 1953 using DBAA − GS10,
        # then drop the auxiliary columns from the feature matrix.
        df_raw = self._extend_credit_spread(df_raw)

        df_raw = self._forward_fill(df_raw)
        df_raw = self._drop_sparse_columns(df_raw)

        df_zscored = df_raw.apply(self._rolling_zscore)
        df_zscored = df_zscored.dropna(how="all")

        if save:
            df_raw.to_csv(self.processed_dir / "raw_monthly.csv")
            df_zscored.to_csv(self.processed_dir / "features_zscored.csv")
            logger.info(f"Saved raw + Z-scored features → {self.processed_dir}")

        logger.info(
            f"Feature matrix: {df_zscored.shape[0]} months × "
            f"{df_zscored.shape[1]} features  "
            f"({df_zscored.index[0].date()} – {df_zscored.index[-1].date()})"
        )
        return df_zscored, df_raw

    def build_returns_matrix(
        self,
        market_data: dict[str, pd.DataFrame],
        assets: list[str],
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Build monthly simple returns for each portfolio asset.

        Returns:
            DataFrame  date × ticker  of monthly returns.
        """
        price_field = self.cfg["data"]["market"].get("price_field", "Close")
        series_dict: dict[str, pd.Series] = {}

        for ticker in assets:
            if ticker not in market_data:
                logger.warning(f"  {ticker} not found in market data — skipped.")
                continue
            prices = market_data[ticker][price_field]
            monthly_px = prices.resample(self.resample_freq).last()
            series_dict[ticker] = monthly_px.pct_change()

        df_returns = pd.DataFrame(series_dict)
        df_returns = df_returns.dropna(how="all")

        if save:
            df_returns.to_csv(self.processed_dir / "asset_returns.csv")
            logger.info(f"Saved asset returns ({df_returns.shape}) → {self.processed_dir}")

        return df_returns

    # ── private ──────────────────────────────────────────────────────────

    def _resample(self, series: pd.Series, name: str) -> pd.Series:
        freq = self.series_meta.get(name, {}).get("freq", "M")

        if freq in ("D", "W", "M"):
            # Take the last available observation in each calendar month
            return series.resample(self.resample_freq).last()

        if freq == "Q":
            # Step 1 — put quarterly values on a monthly grid (NaN in between)
            monthly = series.resample(self.resample_freq).last()
            # Step 2 — interpolate within a max of max_fill months
            monthly = monthly.interpolate(
                method=self.interp_method,
                limit=self.max_fill,
                limit_direction="forward",
            )
            return monthly

        # Fallback
        return series.resample(self.resample_freq).last()

    def _forward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill remaining NaN gaps, capped at max_fill months."""
        return df.ffill(limit=self.max_fill)

    def _extend_credit_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extend credit_spread_baa (BAA10Y on FRED, starts 1986-01-02) back
        to 1953 using the synthetic spread: baa_yield (DBAA) − gs10_yield (GS10).

        The auxiliary columns baa_yield and gs10_yield are dropped after stitching.
        If either auxiliary column is absent, this is a no-op.
        """
        _AUX = ["baa_yield", "gs10_yield"]
        if not all(c in df.columns for c in _AUX):
            return df

        synthetic = df["baa_yield"] - df["gs10_yield"]
        synthetic.name = "credit_spread_baa"

        if "credit_spread_baa" in df.columns:
            # Fill NaN holes in the official series with the synthetic spread
            df["credit_spread_baa"] = df["credit_spread_baa"].combine_first(synthetic)
            logger.info(
                f"  [prep] credit_spread_baa extended: "
                f"{df['credit_spread_baa'].first_valid_index().date()} onwards "
                f"(was {df['credit_spread_baa'].dropna().index[0].date()} with FRED only)"
            )
        else:
            df["credit_spread_baa"] = synthetic

        df = df.drop(columns=_AUX)
        return df

    def _drop_sparse_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop any column where the fraction of *internal* NaN exceeds the
        threshold.  Leading NaN (before a series starts) are excluded from
        the count — they are expected when pulling long history and different
        FRED series start at different dates.
        """
        to_drop = []
        for col in df.columns:
            first_valid = df[col].first_valid_index()
            if first_valid is None:
                to_drop.append(col)   # entirely empty — always drop
                continue
            # Count NaN only from first valid observation onwards
            tail = df.loc[first_valid:, col]
            internal_missing = tail.isna().mean()
            if internal_missing > self.missing_threshold:
                to_drop.append(col)

        if to_drop:
            logger.warning(
                f"Dropping sparse columns "
                f"(>{self.missing_threshold:.0%} internal NaN): {to_drop}"
            )
            df = df.drop(columns=to_drop)
        return df

    def _rolling_zscore(self, s: pd.Series) -> pd.Series:
        """(x − rolling_mean) / rolling_std with configured window."""
        mu = s.rolling(window=self.zscore_window, min_periods=self.zscore_min_periods).mean()
        sigma = s.rolling(window=self.zscore_window, min_periods=self.zscore_min_periods).std()
        # Replace zero-std with NaN to avoid division artefacts
        return (s - mu) / sigma.replace(0.0, np.nan)
