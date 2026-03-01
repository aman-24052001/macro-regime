from __future__ import annotations
"""
model/hmm_trainer.py  —  M in OSEMN (walk-forward training)
-------------------------------------------------------------
Expanding-window walk-forward HMM training with ZERO look-ahead bias.

Protocol
────────
  For each month T in the OOS window:
    1. Train HMM on ALL data before T  (expanding window)
    2. Predict P(regime | data up to T) using the forward algorithm
    3. Store probability vector for month T

  The resulting regime_probs DataFrame is indexed by date and has columns:
    prob_0, prob_1, ..., prob_{K-1}, regime (argmax label)

  This is then fed into the portfolio allocator — which also uses a
  one-month lag (weights set at T are applied to T+1 returns).

Auto-labelling
──────────────
  After the full training window fit, emission means are inspected to
  assign human-readable labels (expansion / stagnation / contraction)
  based on yield-curve feature values — or falls back to numbered labels.
"""

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")

logger = logging.getLogger(__name__)


class HMMTrainer:
    """Walk-forward HMM training with expanding window."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        hmm_cfg = cfg["model"]["hmm"]
        wf_cfg = cfg["model"]["walk_forward"]

        self.cov_type: str = hmm_cfg["covariance_type"]
        self.n_iter: int = hmm_cfg["n_iter"]
        # Fewer restarts during walk-forward for speed
        self.n_fits: int = hmm_cfg.get("walk_forward_n_fits", 10)
        self.tol: float = hmm_cfg["convergence_tol"]
        self.initial_train_years: int = wf_cfg["initial_train_years"]
        self.seed: int = cfg["project"]["random_seed"]

        self.models_dir = Path(cfg["outputs"]["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Features to use — null = all columns
        self.features_for_hmm: list[str] | None = hmm_cfg.get("features_for_hmm")

    # ── public ───────────────────────────────────────────────────────────

    def walk_forward(
        self,
        features: pd.DataFrame,
        n_states: int,
        oos_start: str,
    ) -> pd.DataFrame:
        """
        Expanding-window walk-forward prediction.

        Args:
            features  : full feature matrix (dates × features), may contain NaN
            n_states  : number of HMM states (from BIC selection)
            oos_start : first date for which we produce an OOS prediction

        Returns:
            DataFrame indexed by date with columns:
              prob_0 ... prob_{K-1}  — regime probabilities
              regime               — argmax state label (int)
        """
        # Select feature subset if configured
        X_all = self._select_features(features)

        oos_dt = pd.Timestamp(oos_start)
        oos_idx = X_all[X_all.index >= oos_dt].index

        logger.info(
            f"Walk-forward HMM (K={n_states}): "
            f"OOS {oos_start} → {oos_idx[-1].date()}  "
            f"({len(oos_idx)} months)"
        )

        probs_list: list[np.ndarray] = []
        valid_dates: list[pd.Timestamp] = []

        for i, date in enumerate(oos_idx):
            train = X_all[X_all.index < date].dropna()

            if len(train) < self.initial_train_years * 12:
                logger.debug(f"  {date.date()}: insufficient training data — skip")
                continue

            # Log progress annually
            if i % 12 == 0:
                logger.info(
                    f"  [{i+1}/{len(oos_idx)}] {date.strftime('%Y-%m')} "
                    f"training on {len(train)} months ..."
                )

            model = self._fit_best(train.values, n_states)
            if model is None:
                logger.warning(f"  {date.date()}: all fits failed — skipped")
                continue

            obs = X_all.loc[[date]].dropna()
            if obs.empty or obs.shape[1] != train.shape[1]:
                continue

            try:
                p = model.predict_proba(obs.values)   # shape (1, K)
                probs_list.append(p[0])
                valid_dates.append(date)
            except Exception as exc:
                logger.debug(f"  {date.date()}: predict failed — {exc}")

        prob_cols = [f"prob_{k}" for k in range(n_states)]
        df_probs = pd.DataFrame(probs_list, index=valid_dates, columns=prob_cols)
        df_probs["regime"] = np.argmax(df_probs[prob_cols].values, axis=1)

        logger.info(
            f"Walk-forward complete: {len(df_probs)} months with predictions "
            f"({len(oos_idx) - len(df_probs)} skipped)"
        )
        return df_probs

    def fit_full(
        self,
        features: pd.DataFrame,
        n_states: int,
        name: str = "hmm_full",
    ) -> hmm.GaussianHMM:
        """
        Train a single HMM on the complete dataset (for final analysis/interpretation).
        Saves model to disk.
        """
        X = self._select_features(features).dropna().values
        logger.info(f"Fitting full HMM (K={n_states}) on {len(X)} observations ...")
        model = self._fit_best(X, n_states)
        if model is None:
            raise RuntimeError("Full HMM fit failed for all random seeds.")
        self.save(model, name)
        return model

    def label_regimes(
        self,
        model: hmm.GaussianHMM,
        feature_names: list[str],
        n_states: int,
    ) -> list[str]:
        """
        Auto-label states from emission means.
        Uses yield curve slope: high slope → expansion, low/negative → contraction.
        Falls back to numbered labels if yield curve feature not present.
        """
        means = model.means_   # (K, n_features)

        # Find yield curve column index
        yc_idx = next(
            (i for i, f in enumerate(feature_names) if "yield_curve_10y2y" in f or "yc_slope" in f),
            None,
        )

        if yc_idx is not None and n_states == 3:
            yc_vals = means[:, yc_idx]
            order = np.argsort(yc_vals)   # ascending: most negative = contraction
            labels = [""] * n_states
            labels[order[0]] = "contraction"
            labels[order[1]] = "stagnation"
            labels[order[2]] = "expansion"
            logger.info(f"Regime auto-labels: {labels}")
            return labels

        # Generic fallback
        labels = [f"regime_{i}" for i in range(n_states)]
        logger.warning(
            "Could not auto-label regimes (yield curve feature not found). "
            f"Using: {labels}"
        )
        return labels

    def save(self, model: hmm.GaussianHMM, name: str) -> None:
        path = self.models_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved → {path}")

    def load(self, name: str) -> hmm.GaussianHMM:
        path = self.models_dir / f"{name}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── private ──────────────────────────────────────────────────────────

    def _fit_best(self, X: np.ndarray, k: int) -> hmm.GaussianHMM | None:
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
            except Exception:
                pass
        return best_model

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Respect config override for feature subset."""
        if self.features_for_hmm:
            available = [f for f in self.features_for_hmm if f in features.columns]
            missing = [f for f in self.features_for_hmm if f not in features.columns]
            if missing:
                logger.warning(f"  Configured HMM features not found: {missing}")
            return features[available]
        return features
