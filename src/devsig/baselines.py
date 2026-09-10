"""Raw-data baselines for RQ2. Each has fit(train_runs) and score(run) -> Series over output metrics.
All are wrapped by the same NullModel / threshold logic as the signatures, so the comparison is fair."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest

from .signature import add_history, split_metrics


class Envelope:
    """Fraction of snapshots outside the per-time-index min/max envelope of the training runs (widened by a margin)."""
    def __init__(self, margin: float = 0.05):
        self.margin = margin

    def fit(self, runs):
        _, self.outputs = split_metrics(runs[0].columns)
        n = min(len(r) for r in runs)
        stack = np.stack([r[self.outputs].to_numpy()[:n] for r in runs])
        lo, hi = stack.min(axis=0), stack.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        self.lo, self.hi, self.n = lo - self.margin * span, hi + self.margin * span, n
        return self

    def score(self, run):
        x = run[self.outputs].to_numpy()[: self.n]
        m = min(len(x), self.n)
        out = ((x[:m] < self.lo[:m]) | (x[:m] > self.hi[:m])).mean(axis=0)
        return pd.Series(out, index=self.outputs)


class KS:
    """Two-sample Kolmogorov-Smirnov statistic per output metric against pooled training values."""
    def fit(self, runs):
        _, self.outputs = split_metrics(runs[0].columns)
        self.pool = {o: np.concatenate([r[o].to_numpy() for r in runs]) for o in self.outputs}
        return self

    def score(self, run):
        return pd.Series({o: ks_2samp(run[o].to_numpy(), self.pool[o]).statistic for o in self.outputs})


class RegressionResidual:
    """DARIO-style: gradient-boosted regressor per output on raw continuous inputs (+ history features);
    score = RMSE of the run divided by training RMSE."""
    def __init__(self, lags_s=(0.5, 1.0, 2.0, 5.0), since_change=True, max_iter=100):
        self.lags_s, self.since_change, self.max_iter = lags_s, since_change, max_iter

    def _prep(self, run):
        if any("@" in c for c in run.columns):
            return run  # already prepared
        return add_history(run, self.inputs, self.lags_s, self.since_change)

    def fit(self, runs):
        self.inputs, self.outputs = split_metrics(runs[0].columns)
        df = pd.concat([self._prep(r) for r in runs], ignore_index=True)
        self.features = [c for c in df.columns if c in self.inputs or "@" in c]
        X = df[self.features].to_numpy()
        self.models, self.train_rmse = {}, {}
        for o in self.outputs:
            y = df[o].to_numpy()
            m = HistGradientBoostingRegressor(max_iter=self.max_iter, random_state=0).fit(X, y)
            self.models[o] = m
            self.train_rmse[o] = max(float(np.sqrt(np.mean((m.predict(X) - y) ** 2))), 1e-9)
        return self

    def score(self, run):
        d = self._prep(run)
        X = d[self.features].to_numpy()
        return pd.Series({o: float(np.sqrt(np.mean((self.models[o].predict(X) - d[o].to_numpy()) ** 2))) / self.train_rmse[o] for o in self.outputs})

    def score_windowed(self, run, window_s: float, time_col: str = "simulation/sim-time-sec") -> pd.DataFrame:
        """Residual ratio per window (rows) and output (columns)."""
        d = self._prep(run)
        X = d[self.features].to_numpy()
        t = run[time_col].to_numpy(); w = np.floor((t - t[0]) / window_s).astype(int)
        err = pd.DataFrame({o: (self.models[o].predict(X) - d[o].to_numpy()) ** 2 for o in self.outputs}); err["_w"] = w
        rmse = np.sqrt(err.groupby("_w").mean())
        return rmse / pd.Series(self.train_rmse)


class IForest:
    """Isolation forest per output on (inputs + that output); score = mean anomaly score of the run's snapshots."""
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators

    def fit(self, runs):
        self.inputs, self.outputs = split_metrics(runs[0].columns)
        df = pd.concat(runs, ignore_index=True)
        self.models = {o: IsolationForest(n_estimators=self.n_estimators, random_state=0).fit(df[self.inputs + [o]].to_numpy()) for o in self.outputs}
        return self

    def score(self, run):
        return pd.Series({o: float(-self.models[o].score_samples(run[self.inputs + [o]].to_numpy()).mean()) for o in self.outputs})
