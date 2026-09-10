"""RQ3 sensitivity on all Tier A aircraft: number of baseline runs (detector and dynamic signature),
number of levels K and leaf size (dynamic signature). Uses cached runs; analysis only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
b6 = importlib.import_module("06_tier_a_battery")
b7 = importlib.import_module("07_dynamic_signatures")
from devsig.manoeuvres import BATTERY  # noqa: E402
from devsig.signature import Signatures  # noqa: E402
from devsig.baselines import RegressionResidual  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402
from devsig.evaluate import prf, auc  # noqa: E402

FAULTS = ("inertia_x1.6", "cmq_x0.5", "cmalpha_x0.7", "clo_+0.1")
SETTINGS = [  # (label, layer, kwargs, n_train, sample step: 1 = 10 Hz, 2 = 5 Hz, 5 = 2 Hz)
    ("det_n10", "det", {}, 10, 1), ("det_n20", "det", {}, 20, 1), ("det_n5", "det", {}, 5, 1),
    ("det_lags_1", "det", dict(lags_s=(1.0,)), 20, 1), ("det_lags_long", "det", dict(lags_s=(0.5, 1.0, 2.0, 5.0, 10.0)), 20, 1),
    ("det_5hz", "det", {}, 20, 2), ("det_2hz", "det", {}, 20, 5),
    ("sig_K3", "sig", dict(k=3), 20, 1), ("sig_K5", "sig", dict(k=5), 20, 1), ("sig_K7", "sig", dict(k=7), 20, 1),
    ("sig_leaf50", "sig", dict(k=5, min_samples_leaf=50), 20, 1), ("sig_n10", "sig", dict(k=5), 10, 1),
    ("sig_lags_1", "sig", dict(lags_s=(1.0,)), 20, 1), ("sig_lags_long", "sig", dict(lags_s=(0.5, 1.0, 2.0, 5.0, 10.0)), 20, 1),
    ("sig_equal_width", "sig", dict(scheme="equal_width"), 20, 1), ("sig_5hz", "sig", {}, 20, 2), ("sig_2hz", "sig", {}, 20, 5),
]


def subsample(runs, step):
    return runs if step == 1 else [r.iloc[::step].reset_index(drop=True) for r in runs]


def one_pair(args):
    ac, man = args
    rows = []
    base = b7.load(ac, "baseline", man, range(40))
    if len(base) < 30:
        return rows
    held = base[20:]
    cands = {f: b7.load(ac, f, man, range(100, 110)) for f in FAULTS}
    for label, layer, kw, n_train, step in SETTINGS:
        train = subsample(base[:n_train], step); held_s = subsample(held, step)
        if layer == "det":
            model = RegressionResidual(max_iter=60, **kw).fit(train); score = model.score
        else:
            kws = dict(k=5, min_samples_leaf=10, lags_s=(0.5, 1.0, 2.0, 5.0), since_change=True); kws.update(kw)
            model = Signatures(**kws).fit(train); score = model.vsd
        null = NullModel(pd.DataFrame([score(r) for r in held_s])); thr = null.threshold("p95")
        for f, cand in cands.items():
            if len(cand) < 8:
                continue
            rsd, fl = null.flag_perm(pd.DataFrame([score(r) for r in subsample(cand, step)]))[:2]
            orc = pd.Series({m: (m in b6.ORACLE[f.split("_")[0]]) for m in fl.index})
            m = prf(fl, orc)
            rows.append(dict(aircraft=ac, manoeuvre=man, setting=label, fault=f, recall=m["recall"], precision=m["precision"], auc=auc(rsd, orc),
                             base_score=float(null.vsd_base.mean())))
    print(ac, man, "done", flush=True)
    return rows


def main():
    aircraft = sys.argv[1:]
    pairs = [(ac, man) for ac in aircraft for man in BATTERY]
    with Pool(int(os.environ.get("WORKERS", max(1, os.cpu_count() - 1)))) as pool:
        results = pool.map(one_pair, pairs, chunksize=1)
    out = pd.DataFrame([r for rs in results for r in rs])
    out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "17_sensitivity_rq3.csv"), index=False)
    print("\n== mean recall / precision / AUC per setting (all aircraft, manoeuvres, faults)")
    print(out.groupby("setting")[["recall", "precision", "auc", "base_score"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
