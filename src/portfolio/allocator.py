from __future__ import annotations
"""
portfolio/allocator.py  —  N in OSEMN (interpret / apply)
-----------------------------------------------------------
Converts HMM regime probabilities into portfolio weights using
probability-weighted blending of regime-specific target portfolios.

Formula:
  w_T = P(expansion_T) × w_exp  +
        P(stagnation_T) × w_stag +
        P(contraction_T) × w_cont

This produces smooth, gradual weight shifts — not binary on/off switching.
Weights are normalised to sum to 1 and capped by risk management rules.

All regime portfolio definitions and risk limits come from config.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroAllocator:
    """Compute portfolio weights from HMM regime probabilities."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        port_cfg = cfg["portfolio"]

        self.regime_portfolios: dict[str, dict[str, float]] = port_cfg["regime_portfolios"]
        self.max_single_weight: float = port_cfg["risk_management"]["max_single_weight"]
        self.vol_scale_target: float | None = port_cfg["risk_management"]["vol_scale_target"]
        self.vol_scale_cap: float = port_cfg["risk_management"]["vol_scale_cap"]

        self.assets: list[str] = list(cfg["data"]["market"]["etfs"].keys())

    # ── public ───────────────────────────────────────────────────────────

    def compute_weights(
        self,
        regime_probs: pd.DataFrame,
        regime_names: list[str],
    ) -> pd.DataFrame:
        """
        Blend regime-specific portfolios by probability weights.

        Args:
            regime_probs  : DataFrame with columns  prob_0, prob_1, ...  (sums to 1 per row)
            regime_names  : list mapping state index → regime name
                            e.g. ["expansion", "stagnation", "contraction"]

        Returns:
            DataFrame  (dates × assets)  of portfolio weights.
        """
        prob_cols = [c for c in regime_probs.columns if c.startswith("prob_")]

        weights_list: list[dict] = []
        for _, row in regime_probs.iterrows():
            w = {asset: 0.0 for asset in self.assets}

            for k, prob_col in enumerate(prob_cols):
                prob_k = row.get(prob_col, 0.0)
                if k >= len(regime_names):
                    continue
                regime_name = regime_names[k]
                target = self.regime_portfolios.get(regime_name, {})

                for asset in self.assets:
                    w[asset] += prob_k * target.get(asset, 0.0)

            # Hard cap per asset
            w = {a: min(v, self.max_single_weight) for a, v in w.items()}

            # Normalise to sum = 1
            total = sum(w.values())
            if total > 1e-9:
                w = {a: v / total for a, v in w.items()}

            weights_list.append(w)

        df_weights = pd.DataFrame(weights_list, index=regime_probs.index)
        # Ensure consistent column order
        df_weights = df_weights[[a for a in self.assets if a in df_weights.columns]]

        logger.info(
            f"Weights computed: {len(df_weights)} months  "
            f"avg SPY={df_weights.get('SPY', pd.Series()).mean():.1%}  "
            f"avg TLT={df_weights.get('TLT', pd.Series()).mean():.1%}"
        )
        return df_weights

    def apply_vol_scaling(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """
        Optional inverse-volatility scaling (enabled when config
        risk_management.vol_scale_target is not null).

        For each asset:
          w_adj = w_target × (σ_target / σ_realized)  capped at vol_scale_cap × w_target
        """
        if self.vol_scale_target is None:
            logger.info("Vol scaling disabled (vol_scale_target is null in config).")
            return weights

        scaled = weights.copy()
        for asset in self.assets:
            if asset not in returns.columns:
                continue
            sigma_r = returns[asset].rolling(window).std() * np.sqrt(12)
            sigma_r = sigma_r.reindex(weights.index).ffill().clip(lower=0.01)
            scale = (self.vol_scale_target / sigma_r).clip(upper=self.vol_scale_cap)
            scaled[asset] = (weights[asset] * scale).clip(upper=self.max_single_weight)

        # Re-normalise row sums
        row_sums = scaled.sum(axis=1).replace(0, 1)
        scaled = scaled.div(row_sums, axis=0)
        return scaled

    def static_weights(self, weights_dict: dict) -> pd.DataFrame:
        """
        Helper: build a constant weight DataFrame for benchmark use.
        Fills the given weights across all index dates (to be aligned externally).
        """
        return pd.DataFrame(
            [weights_dict],
            columns=self.assets,
        ).fillna(0.0)

    def inverse_vol_static(
        self,
        returns: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Static inverse-volatility portfolio — recomputed each month.
        Used as an additional benchmark.
        """
        sigma = returns.rolling(window).std() * np.sqrt(12)
        inv_sigma = 1.0 / sigma.clip(lower=0.01)
        row_sums = inv_sigma.sum(axis=1)
        inv_vol_weights = inv_sigma.div(row_sums, axis=0)
        inv_vol_weights.columns.name = None
        return inv_vol_weights[[a for a in self.assets if a in inv_vol_weights.columns]]
