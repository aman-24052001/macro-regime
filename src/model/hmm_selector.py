from __future__ import annotations
"""
model/hmm_selector.py  —  M in OSEMN (model selection step)
------------------------------------------------------------
Data-driven selection of the optimal number of HMM hidden states
using the Bayesian Information Criterion (BIC).

This runs BEFORE the walk-forward trainer.  The EDA notebook calls
this module, examines the elbow chart and summary table, then the
chosen K is written back to config (or passed directly to HMMTrainer).

Design:
  • K search range comes from config['model']['hmm']['n_states_range']
  • Multiple random restarts avoid EM local optima
  • BIC penalises model complexity — prevents over-fitting more than AIC
  • Full BIC formula for Gaussian HMM (exact parameter count)
"""

import logging
import warnings

import numpy as np
from hmmlearn import hmm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")

logger = logging.getLogger(__name__)


class HMMSelector:
    """Select optimal K states via BIC on the training window."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        hmm_cfg = cfg["model"]["hmm"]

        lo, hi = hmm_cfg["n_states_range"]
        self.k_range = range(lo, hi + 1)
        self.force_k: int | None = hmm_cfg.get("force_k")   # economic override
        self.cov_type: str = hmm_cfg["covariance_type"]
        self.n_iter: int = hmm_cfg["n_iter"]
        self.n_fits: int = hmm_cfg["n_fits"]       # BIC phase — use full restarts
        self.tol: float = hmm_cfg["convergence_tol"]
        self.seed: int = cfg["project"]["random_seed"]

    # ── public ───────────────────────────────────────────────────────────

    def select(self, X: np.ndarray) -> dict:
        """
        Fit HMMs for each K in k_range and select by BIC.

        Args:
            X : feature matrix  (n_samples, n_features)  — no NaN.

        Returns:
            dict with keys:
              best_k        — recommended number of states
              best_model    — trained hmmlearn model for best_k
              bic_scores    — {k: bic}
              aic_scores    — {k: aic}
              all_results   — full per-K diagnostics
        """
        results: dict[int, dict] = {}

        for k in self.k_range:
            logger.info(f"  [BIC] K={k}: fitting {self.n_fits} random starts ...")
            model, log_lik = self._fit_best(X, k)

            if model is None:
                logger.warning(f"  [BIC] K={k}: all fits failed — skipped.")
                continue

            n_params = self._count_params(k, X.shape[1])
            bic = self._bic(log_lik, n_params, len(X))
            aic = self._aic(log_lik, n_params)

            results[k] = {
                "model": model,
                "log_likelihood": round(log_lik, 2),
                "n_params": n_params,
                "bic": round(bic, 2),
                "aic": round(aic, 2),
            }
            logger.info(
                f"  [BIC] K={k}: log-lik={log_lik:.1f}  "
                f"BIC={bic:.1f}  AIC={aic:.1f}"
            )

        if not results:
            raise RuntimeError("All HMM fits failed.  Check data quality and feature count.")

        bic_best_k = min(results, key=lambda k: results[k]["bic"])

        if self.force_k is not None and self.force_k in results:
            best_k = self.force_k
            logger.info(
                f"  [BIC] → BIC best K={bic_best_k}  "
                f"(BIC={results[bic_best_k]['bic']:.1f}), "
                f"but OVERRIDING with force_k={best_k} (economic prior)"
            )
        else:
            best_k = bic_best_k
            logger.info(
                f"  [BIC] → Selected K={best_k}  "
                f"(BIC={results[best_k]['bic']:.1f})"
            )

        return {
            "best_k": best_k,
            "best_model": results[best_k]["model"],
            "bic_scores": {k: v["bic"] for k, v in results.items()},
            "aic_scores": {k: v["aic"] for k, v in results.items()},
            "all_results": results,
        }

    # ── private ──────────────────────────────────────────────────────────

    def _fit_best(self, X: np.ndarray, k: int) -> tuple:
        """Multiple random restarts → return (best_model, best_log_likelihood)."""
        best_score = -np.inf
        best_model = None

        for i in range(self.n_fits):
            try:
                model = hmm.GaussianHMM(
                    n_components=k,
                    covariance_type=self.cov_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=self.seed + i,
                )
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as exc:
                logger.debug(f"    fit #{i} K={k} failed: {exc}")

        return best_model, best_score

    @staticmethod
    def _count_params(k: int, n_features: int) -> int:
        """
        Exact parameter count for Gaussian HMM with full covariance:
          transition matrix A   : k*(k-1)   (each row sums to 1, so k-1 free)
          initial distribution  : k-1
          emission means        : k * n_features
          emission covariances  : k * n_features*(n_features+1)/2  (symmetric)
        """
        return (
            k * (k - 1)
            + (k - 1)
            + k * n_features
            + k * n_features * (n_features + 1) // 2
        )

    @staticmethod
    def _bic(log_lik: float, n_params: int, n_samples: int) -> float:
        return -2.0 * log_lik + n_params * np.log(n_samples)

    @staticmethod
    def _aic(log_lik: float, n_params: int) -> float:
        return -2.0 * log_lik + 2.0 * n_params
