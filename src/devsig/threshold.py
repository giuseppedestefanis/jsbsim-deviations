"""Null distribution of RSD from held-out baseline runs, and thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd


class NullModel:
    def __init__(self, held_out_vsd: pd.DataFrame, n_cand: int = 10, n_boot: int = 500):
        """held_out_vsd: rows = held-out baseline runs, columns = output metrics (VSD values).

        A candidate version is judged by the median score of its n_cand runs, so the threshold is drawn from the
        distribution of that statistic: medians of n_boot bootstrap samples (n_cand held-out runs drawn with
        replacement: pseudo-candidates). Drawing without replacement from the held-out set under-disperses the
        statistic and roughly doubles the false-alarm rate. n_cand=None falls back to single held-out runs."""
        self.vsd = held_out_vsd
        self.vsd_base = held_out_vsd.mean(axis=0)
        self.rsd_null = held_out_vsd - self.vsd_base
        self.n_cand, self.n_boot = n_cand, n_boot
        if n_cand and len(held_out_vsd) > n_cand:
            rng = np.random.default_rng(0)
            idx = [rng.choice(len(held_out_vsd), size=n_cand, replace=True) for _ in range(n_boot)]
            self.rsd_null_stat = pd.DataFrame([self.rsd_null.iloc[i].median(axis=0) for i in idx])
        else:
            self.rsd_null_stat = self.rsd_null

    def rsd(self, vsd_candidate: pd.Series | pd.DataFrame):
        return vsd_candidate - self.vsd_base

    def flag_perm(self, cand_scores: pd.DataFrame, alpha: float = 0.05, n_perm: int = 5000):
        """Decide per output with a one-sided permutation test of the candidate runs' median score against
        the held-out runs (statistic: median(candidate) - mean(held-out)). Returns (rsd, flagged, p-value):
        rsd is the observed statistic (used for ranking), flagged is p < alpha."""
        p = _perm_test(self.vsd, cand_scores, n_perm)
        rsd = cand_scores.reindex(columns=self.vsd.columns).median(axis=0) - self.vsd_base
        return rsd, p < alpha, p

    def threshold(self, method: str = "p95", margin: float = 0.0) -> pd.Series:
        if method == "max":
            t = self.rsd_null_stat.max(axis=0)
        elif method.startswith("p"):
            t = self.rsd_null_stat.quantile(float(method[1:]) / 100.0, axis=0)
        elif method == "3sigma":
            t = 3.0 * self.rsd_null_stat.std(axis=0)
        else:
            raise ValueError(method)
        return t + margin


def _perm_test(held: pd.DataFrame, cand: pd.DataFrame, n_perm: int, seed: int = 0) -> pd.Series:
    """One-sided permutation p-value per output for the statistic median(candidate) - mean(held-out), under
    exchangeability of the candidate and held-out runs (both unseen by the model)."""
    H = held.to_numpy(dtype=float); C = cand.reindex(columns=held.columns).to_numpy(dtype=float)
    pooled = np.concatenate([C, H]); nc, n = len(C), len(pooled)
    obs = np.median(C, axis=0) - H.mean(axis=0)
    rng = np.random.default_rng(seed)
    count = np.ones(pooled.shape[1])
    for _ in range(n_perm):
        idx = rng.permutation(n)
        stat = np.median(pooled[idx[:nc]], axis=0) - pooled[idx[nc:]].mean(axis=0)
        count += stat >= obs - 1e-12
    return pd.Series(count / (n_perm + 1), index=held.columns)


def flag(rsd_candidate_runs: pd.DataFrame, thr: pd.Series, agg: str = "median") -> tuple[pd.Series, pd.Series]:
    """Aggregate candidate-run RSDs per metric and compare with thresholds. Returns (rsd_agg, flagged)."""
    r = rsd_candidate_runs.median(axis=0) if agg == "median" else rsd_candidate_runs.mean(axis=0)
    return r, r > thr
