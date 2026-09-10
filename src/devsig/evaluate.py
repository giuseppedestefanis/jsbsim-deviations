"""Precision / recall / AUC at metric and category level against an oracle."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .signature import category


def prf(flagged: pd.Series, oracle: pd.Series) -> dict:
    f = flagged.astype(bool); o = oracle.reindex(f.index).fillna(False).astype(bool)
    tp = int((f & o).sum()); fp = int((f & ~o).sum()); fn = int((~f & o).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r,
                f1=(2 * p * r / (p + r)) if (p + r) and not np.isnan(p) else float("nan"))


def to_category(series: pd.Series, how="any") -> pd.Series:
    g = series.groupby(series.index.map(category))
    return g.any() if how == "any" else g.max()


def auc(score: pd.Series, oracle: pd.Series) -> float:
    o = oracle.reindex(score.index).fillna(False).astype(int)
    if o.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(o, score.fillna(score.min())))
