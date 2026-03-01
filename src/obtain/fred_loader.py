from __future__ import annotations
"""
obtain/fred_loader.py  —  O in OSEMN
--------------------------------------
Downloads macro time-series from FRED via the fredapi library.

All series IDs, dates, and cache paths come from config — nothing
is hardcoded here. Supports force-refresh to bypass local cache.

Key design:
  - Each series cached as <cache_dir>/fred/<logical_name>.csv
  - Returns a dict[name → pd.Series] with original FRED frequency
  - Downstream preprocessor handles resampling to monthly
"""

import logging
from pathlib import Path

import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)


class FREDLoader:
    """Download and cache FRED macro series declared in config."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        fred_cfg = cfg["data"]["fred"]

        self.fred = Fred(api_key=fred_cfg["api_key"])
        self.start_date: str = fred_cfg["start_date"]
        self.end_date: str | None = fred_cfg.get("end_date")  # None → today
        self.series_config: dict = fred_cfg["series"]

        cache_root = Path(cfg["outputs"]["data_cache_dir"])
        self.cache_dir = cache_root / "fred"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── public ───────────────────────────────────────────────────────────

    def load(self, force_refresh: bool = False) -> dict[str, pd.Series]:
        """
        Return all configured FRED series.

        Args:
            force_refresh: if True, ignore cache and re-download.

        Returns:
            dict mapping logical name → pd.Series (at original FRED frequency).
        """
        results: dict[str, pd.Series] = {}

        for name, meta in self.series_config.items():
            cache_path = self.cache_dir / f"{name}.csv"

            if cache_path.exists() and not force_refresh:
                logger.info(f"  [FRED] {name:30s} <- cache ({cache_path.name})")
                series = self._read_cache(cache_path, name)
            else:
                logger.info(
                    f"  [FRED] {name:30s} <- downloading {meta['id']} "
                    f"({meta.get('freq','?')}) ..."
                )
                series = self._download(name, meta)
                self._write_cache(series, cache_path)

            results[name] = series

        logger.info(f"FRED: {len(results)} series loaded.")
        return results

    def describe(self) -> pd.DataFrame:
        """Return a summary table of all configured series (useful in EDA notebook)."""
        rows = []
        for name, meta in self.series_config.items():
            rows.append(
                {
                    "logical_name": name,
                    "fred_id": meta["id"],
                    "frequency": meta.get("freq", "?"),
                    "description": meta.get("description", ""),
                }
            )
        return pd.DataFrame(rows)

    # ── private ──────────────────────────────────────────────────────────

    def _download(self, name: str, meta: dict) -> pd.Series:
        kwargs = {"observation_start": self.start_date}
        if self.end_date:
            kwargs["observation_end"] = self.end_date

        raw = self.fred.get_series(meta["id"], **kwargs)
        raw.name = name
        raw.index = pd.to_datetime(raw.index)
        return raw

    def _write_cache(self, series: pd.Series, path: Path) -> None:
        series.to_frame().to_csv(path)
        logger.debug(f"    cached → {path}")

    def _read_cache(self, path: Path, name: str) -> pd.Series:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        s = df.squeeze()
        s.name = name
        return s
