from __future__ import annotations
"""
explore/eda_stats.py  —  E in OSEMN
--------------------------------------
All statistical EDA outputs: stationarity, correlations, distributions,
BIC recommendation, regime overlap with NBER.

Crucially, the outputs of this module INFORM model hyperparameters:
  • Which features to include in HMM  (stationarity + correlation)
  • How many HMM states to use        (BIC recommendation)
  • Regime quality                    (NBER latency analysis)

Nothing from this module is hardcoded — thresholds come from config.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

# KPSS p-value clamped at table boundary — not an error, just cosmetic noise
warnings.filterwarnings("ignore", category=InterpolationWarning)

logger = logging.getLogger(__name__)


class EDAStats:
    """Statistical summaries to guide model design decisions."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        expl = cfg["exploration"]
        self.stationarity_alpha: float = expl["stationarity_alpha"]
        self.corr_method: str = expl["correlation_method"]
        self.high_corr_threshold: float = expl["high_corr_threshold"]
        self.nber_recessions: list[dict] = expl["nber_recessions"]

    # ── stationarity ─────────────────────────────────────────────────────

    def stationarity_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ADF + KPSS stationarity tests for each feature column.

        Decision rule (Kwiatkowski-Phillips-Schmidt-Shin cross-validation):
          ADF: H0=unit root → small p ⇒ stationary
          KPSS: H0=stationary → large p ⇒ stationary

        Returns DataFrame sorted by ADF p-value (most non-stationary first).
        """
        rows = []
        for col in df.columns:
            s = df[col].dropna()
            if len(s) < 20:
                continue

            adf_stat, adf_p, _, _, adf_cv, _ = adfuller(s, autolag="AIC")

            try:
                kpss_stat, kpss_p, _, kpss_cv = kpss(s, regression="c", nlags="auto")
                kpss_stationary = kpss_p > self.stationarity_alpha
            except Exception:
                kpss_stat, kpss_p, kpss_stationary = np.nan, np.nan, None

            adf_stationary = adf_p < self.stationarity_alpha
            # Both tests agree → higher confidence
            verdict = "STATIONARY" if (adf_stationary and kpss_stationary) \
                else "NON-STATIONARY" if (not adf_stationary and not kpss_stationary) \
                else "AMBIGUOUS"

            rows.append(
                {
                    "feature": col,
                    "adf_stat": round(adf_stat, 3),
                    "adf_p": round(adf_p, 4),
                    "adf_stationary": adf_stationary,
                    "kpss_stat": round(kpss_stat, 3) if not np.isnan(kpss_stat) else np.nan,
                    "kpss_p": round(kpss_p, 4) if not np.isnan(kpss_p) else np.nan,
                    "kpss_stationary": kpss_stationary,
                    "verdict": verdict,
                }
            )

        return pd.DataFrame(rows).sort_values("adf_p", ascending=False)

    def recommended_features_for_hmm(self, stationarity_df: pd.DataFrame) -> list[str]:
        """
        Return feature names that passed stationarity (ADF + KPSS both agree).
        This is the data-driven way to choose HMM input features.
        """
        mask = stationarity_df["verdict"] == "STATIONARY"
        stationary = stationarity_df.loc[mask, "feature"].tolist()
        logger.info(
            f"Stationary features ({len(stationary)}/{len(stationarity_df)}): {stationary}"
        )
        return stationary

    # ── correlations ─────────────────────────────────────────────────────

    def correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.corr(method=self.corr_method)

    def high_correlation_pairs(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        List feature pairs with |correlation| > threshold.
        Helps identify redundant features to drop before HMM.
        """
        rows = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = corr_matrix.iloc[i, j]
                if abs(c) >= self.high_corr_threshold:
                    rows.append({"feature_a": cols[i], "feature_b": cols[j], "corr": round(c, 3)})
        return pd.DataFrame(rows).sort_values("corr", key=abs, ascending=False)

    # ── descriptive stats ────────────────────────────────────────────────

    def descriptive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full descriptive statistics including skewness and kurtosis."""
        desc = df.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
        desc.loc["skewness"] = df.skew()
        desc.loc["kurtosis"] = df.kurt()
        return desc

    def normality_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """Jarque-Bera normality test for each feature."""
        rows = []
        for col in df.columns:
            s = df[col].dropna()
            jb_stat, jb_p = stats.jarque_bera(s)
            rows.append(
                {
                    "feature": col,
                    "jb_stat": round(jb_stat, 3),
                    "jb_p": round(jb_p, 4),
                    "is_normal": jb_p > self.stationarity_alpha,
                }
            )
        return pd.DataFrame(rows).sort_values("jb_p")

    # ── regime vs NBER ───────────────────────────────────────────────────

    def nber_latency_analysis(
        self,
        regime_probs: pd.DataFrame,
        contraction_col: str = "prob_contraction",
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """
        For each NBER recession, measure how many months after the
        official start the model first assigns P(contraction) > threshold.

        Args:
            regime_probs    : DataFrame with a column for contraction probability.
            contraction_col : column name for contraction probability.
            threshold       : detection threshold (default 0.50).

        Returns:
            DataFrame with recession name, NBER start, model detection date,
            and latency in months.
        """
        if contraction_col not in regime_probs.columns:
            # Try to find by position (last state = contraction by convention)
            prob_cols = [c for c in regime_probs.columns if c.startswith("prob_")]
            if prob_cols:
                contraction_col = prob_cols[-1]
                logger.warning(f"  Using '{contraction_col}' as contraction proxy.")
            else:
                logger.error("No probability columns found in regime_probs.")
                return pd.DataFrame()

        rows = []
        for rec in self.nber_recessions:
            start = pd.Timestamp(rec["start"])
            end = pd.Timestamp(rec["end"])

            post = regime_probs.loc[
                (regime_probs.index >= start) & (regime_probs[contraction_col] > threshold),
                contraction_col,
            ]

            if not post.empty:
                detect_date = post.index[0]
                latency = (detect_date.year - start.year) * 12 + (
                    detect_date.month - start.month
                )
            else:
                detect_date = None
                latency = None

            rows.append(
                {
                    "recession": rec["name"],
                    "nber_start": start.strftime("%Y-%m"),
                    "nber_end": end.strftime("%Y-%m"),
                    "model_detects": detect_date.strftime("%Y-%m") if detect_date else "Not detected",
                    "latency_months": latency,
                }
            )

        df = pd.DataFrame(rows)
        if "latency_months" in df.columns:
            valid = df["latency_months"].dropna()
            if len(valid) > 0:
                logger.info(
                    f"  Average detection latency: {valid.mean():.1f} months "
                    f"(target: ≤3 months)"
                )
        return df

    def regime_conditional_stats(
        self, df: pd.DataFrame, regime_labels: pd.Series
    ) -> pd.DataFrame:
        """
        Mean feature value per regime — helps label 'Expansion/Stagnation/Contraction'
        from unlabelled HMM states and understand which features drive each state.
        """
        combined = df.copy()
        combined["_regime"] = regime_labels
        return combined.groupby("_regime").mean().T

    # ── BIC recommendation ───────────────────────────────────────────────

    def bic_recommendation(self, bic_scores: dict) -> dict:
        """
        Summarise BIC selection results.  Call this AFTER HMMSelector.select()
        to get a human-readable explanation to display in the EDA notebook.
        """
        best_k = min(bic_scores, key=bic_scores.get)
        sorted_ks = sorted(bic_scores.keys())
        margin = (
            bic_scores[best_k - 1] - bic_scores[best_k]
            if best_k > sorted_ks[0]
            else None
        )
        return {
            "recommended_k": best_k,
            "bic_scores": {k: round(v, 1) for k, v in bic_scores.items()},
            "margin_vs_simpler": round(margin, 1) if margin is not None else "N/A",
            "reasoning": (
                f"K={best_k} minimises BIC (lower = better fit-complexity tradeoff). "
                f"BIC improvement over K={best_k-1}: "
                f"{margin:.1f} points." if margin else
                f"K={best_k} minimises BIC.  This is the simplest model in the search range."
            ),
        }
