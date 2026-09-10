"""Equal-frequency (and equal-width) discretisation fitted on baseline data."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Discretizer:
    def __init__(self, k: int = 3, scheme: str = "equal_frequency"):
        self.k, self.scheme = k, scheme
        self.edges: dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> "Discretizer":
        for c in df.columns:
            x = df[c].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0 or np.nanmax(x) == np.nanmin(x):
                self.edges[c] = np.array([])  # constant: single bin
                continue
            if self.scheme == "equal_frequency":
                qs = np.quantile(x, np.linspace(0, 1, self.k + 1)[1:-1])
                e = np.unique(qs)
            else:
                e = np.linspace(x.min(), x.max(), self.k + 1)[1:-1]
            self.edges[c] = e
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in df.columns:
            e = self.edges.get(c)
            if e is None:
                raise KeyError(c)
            out[c] = np.searchsorted(e, df[c].to_numpy(dtype=float), side="right") if e.size else np.zeros(len(df), dtype=int)
        return pd.DataFrame(out, index=df.index)
