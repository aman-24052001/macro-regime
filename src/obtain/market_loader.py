from __future__ import annotations
"""
obtain/market_loader.py  —  O in OSEMN
----------------------------------------
Downloads adjusted price history for ETFs and case-study stocks
via yfinance.

All tickers, date ranges, and cache paths come from config.
Returns a dict[ticker → pd.DataFrame] with columns [Close, Volume].
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketLoader:
    """Download and cache adjusted price data from Yahoo Finance."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        market_cfg = cfg["data"]["market"]

        # Tickers from config — never hardcoded
        self.etfs: list[str] = list(market_cfg["etfs"].keys())
        self.stocks: list[str] = list(market_cfg.get("stocks", {}).keys())
        self.price_field: str = market_cfg.get("price_field", "Close")

        # Share date range with FRED for consistency
        self.start_date: str = cfg["data"]["fred"]["start_date"]
        self.end_date: str | None = cfg["data"]["fred"].get("end_date")

        cache_root = Path(cfg["outputs"]["data_cache_dir"])
        self.cache_dir = cache_root / "market"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── public ───────────────────────────────────────────────────────────

    @property
    def all_tickers(self) -> list[str]:
        return self.etfs + self.stocks

    def load(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """
        Return price/volume data for all configured tickers.

        Returns:
            dict  ticker → DataFrame(Close, Volume)  at daily frequency.
        """
        results: dict[str, pd.DataFrame] = {}

        for ticker in self.all_tickers:
            cache_path = self.cache_dir / f"{ticker}.csv"

            if cache_path.exists() and not force_refresh:
                logger.info(f"  [MKT] {ticker:8s} <- cache")
                df = self._read_cache(cache_path)
            else:
                logger.info(f"  [MKT] {ticker:8s} <- downloading ...")
                df = self._download(ticker)
                if df is not None and not df.empty:
                    df.to_csv(cache_path)
                else:
                    logger.warning(f"  [MKT] {ticker}: no data returned, skipping.")
                    continue

            results[ticker] = df

        logger.info(
            f"Market: {len(self.etfs)} ETFs + {len(self.stocks)} stocks loaded."
        )
        return results

    def describe(self) -> pd.DataFrame:
        """Summary of configured market instruments (for EDA notebook)."""
        rows = []
        for t in self.etfs:
            meta = self.cfg["data"]["market"]["etfs"][t]
            rows.append(
                {
                    "ticker": t,
                    "type": "ETF",
                    "asset_class": meta.get("asset_class", ""),
                    "description": meta.get("description", ""),
                }
            )
        for t in self.stocks:
            meta = self.cfg["data"]["market"]["stocks"][t]
            rows.append(
                {
                    "ticker": t,
                    "type": "Stock",
                    "asset_class": "equity",
                    "description": meta.get("description", ""),
                }
            )
        return pd.DataFrame(rows)

    # ── private ──────────────────────────────────────────────────────────

    def _download(self, ticker: str) -> pd.DataFrame | None:
        # yfinance 1.x removed the `progress` kwarg from Ticker.history()
        kwargs = {"start": self.start_date, "auto_adjust": True}
        if self.end_date:
            kwargs["end"] = self.end_date

        try:
            t = yf.Ticker(ticker)
            hist = t.history(**kwargs)
            if hist.empty:
                return None
            # Normalise timezone-aware index -> naive UTC date
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            return hist[["Close", "Volume"]]
        except Exception as exc:
            logger.error(f"  [MKT] {ticker}: download failed — {exc}")
            return None

    def _read_cache(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0, parse_dates=True)
