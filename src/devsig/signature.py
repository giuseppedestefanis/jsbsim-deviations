"""Signatures: one decision tree per output metric, trained on discretised baseline runs. VSD and RSD."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .discretize import Discretizer

INPUT_PREFIXES = ("fcs/", "gear/gear-cmd-norm", "ap/altitude_setpoint", "ap/heading_setpoint", "ap/airspeed_setpoint", "propulsion/cutoff_cmd", "propulsion/magneto_cmd")
OUTPUT_PREFIXES = ("position/", "attitude/", "velocities/", "accelerations/", "aero/alpha-deg", "aero/beta-deg")
EXCLUDE_INPUT_SUBSTR = ("trim-sum", "pos-deg",)  # keep norm/rad positions; drop duplicates in degrees
EXCLUDE_OUTPUTS = ("position/lat-geod-deg", "position/long-gc-deg")  # integrals of the velocities: not scored


def split_metrics(columns) -> tuple[list[str], list[str]]:
    ins = [c for c in columns if c.startswith(INPUT_PREFIXES) and not any(s in c for s in EXCLUDE_INPUT_SUBSTR)]
    outs = [c for c in columns if c.startswith(OUTPUT_PREFIXES) and c not in EXCLUDE_OUTPUTS]
    return ins, outs


CATEGORY = {"position/": "P", "attitude/": "O", "velocities/": "V", "accelerations/": "A", "aero/": "Ae"}


def category(metric: str) -> str:
    for k, v in CATEGORY.items():
        if metric.startswith(k):
            return v
    return "?"


TIME_COL = "simulation/sim-time-sec"


def add_history(run: pd.DataFrame, inputs: list[str], lags_s: tuple, since_change: bool, time_col: str = TIME_COL) -> pd.DataFrame:
    """Dynamic-signature features: lagged copies of each input and, per input, the time since it last changed.
    Lags are in seconds and converted to rows using the run's sampling interval. Edge rows repeat the first value."""
    out = run.copy()
    t = run[time_col].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    for c in inputs:
        x = run[c].to_numpy()
        for L in lags_s:
            n = max(1, int(round(L / dt)))
            out[f"{c}@-{L:g}s"] = np.concatenate([np.repeat(x[0], n), x[:-n]]) if n < len(x) else np.repeat(x[0], len(x))
        if since_change:
            # time since the input last changed; zero until its first change, so an input that never moves
            # carries no clock (otherwise attribution would pick "time since start" of a constant input)
            changed = np.concatenate([[False], np.abs(np.diff(x)) > 1e-6])
            last = pd.Series(np.where(changed, t, np.nan)).ffill().to_numpy()
            out[f"{c}@since"] = np.where(np.isnan(last), 0.0, t - last)
    return out


class Signatures:
    def __init__(self, k: int = 3, scheme: str = "equal_frequency", max_depth: int | None = None,
                 min_samples_leaf: int = 20, criterion: str = "entropy", lags_s: tuple = (), since_change: bool = False):
        self.k, self.scheme = k, scheme
        self.tree_kw = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=0)
        self.lags_s, self.since_change = tuple(lags_s), since_change
        self.disc: Discretizer | None = None
        self.trees: dict[str, DecisionTreeClassifier] = {}
        self.inputs: list[str] = []      # raw input metrics
        self.features: list[str] = []    # inputs + history features
        self.outputs: list[str] = []

    def _prep(self, run: pd.DataFrame) -> pd.DataFrame:
        if self.lags_s or self.since_change:
            if any("@" in c for c in run.columns):
                return run  # already prepared (callers may cache prepared runs)
            return add_history(run, self.inputs, self.lags_s, self.since_change)
        return run

    def fit(self, runs: list[pd.DataFrame]) -> "Signatures":
        self.inputs, self.outputs = split_metrics(runs[0].columns)
        runs = [self._prep(r) for r in runs]
        df = pd.concat(runs, ignore_index=True)
        self.features = [c for c in df.columns if c in self.inputs or "@" in c]
        self.disc = Discretizer(self.k, self.scheme).fit(df[self.features + self.outputs])
        d = self.disc.transform(df[self.features + self.outputs])
        X = d[self.features].to_numpy()
        for o in self.outputs:
            y = d[o].to_numpy()
            t = DecisionTreeClassifier(**self.tree_kw)
            t.fit(X, y)
            self.trees[o] = t
        return self

    def vsd(self, run: pd.DataFrame) -> pd.Series:
        """Version Signature Deviation per output metric on one run: 1 - accuracy."""
        run = self._prep(run)
        d = self.disc.transform(run[self.features + self.outputs])
        X = d[self.features].to_numpy()
        return pd.Series({o: 1.0 - float((self.trees[o].predict(X) == d[o].to_numpy()).mean()) for o in self.outputs})

    def vsd_windowed(self, run: pd.DataFrame, window_s: float, time_col: str = "simulation/sim-time-sec") -> pd.DataFrame:
        """VSD per window (rows) and output metric (columns)."""
        run = self._prep(run)
        d = self.disc.transform(run[self.features + self.outputs])
        X = d[self.features].to_numpy()
        t = run[time_col].to_numpy()
        w = np.floor((t - t[0]) / window_s).astype(int)
        preds = {o: (self.trees[o].predict(X) != d[o].to_numpy()).astype(float) for o in self.outputs}
        err = pd.DataFrame(preds)
        err["_w"] = w
        return err.groupby("_w").mean()

    def n_rules(self) -> pd.Series:
        return pd.Series({o: t.get_n_leaves() for o, t in self.trees.items()})
