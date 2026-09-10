"""RQ4 on Tier A: whole-run vs windowed detector (regression residual) and windowed dynamic signatures.
Windowed score of a run = max over windows of the per-window score relative to the per-window training mean;
null from the same statistic on held-out runs (95th percentile)."""
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

FAULTS = ("inertia_x1.6", "cmq_x0.5", "cmalpha_x0.7", "clo_+0.1", "noop")
WINDOWS = (5.0, 10.0)


def windowed_scores(model, runs, W, kind):
    if kind == "det":
        return [model.score_windowed(r, W) for r in runs]
    return [model.vsd_windowed(r, W) for r in runs]


def one_pair(args):
    ac, man = args
    rows = []
    base = b7.load(ac, "baseline", man, range(40))
    if len(base) < 30:
        return rows
    train, held = base[:len(base) // 2], base[len(base) // 2:]
    cands = {f: b7.load(ac, f, man, range(100, 110)) for f in FAULTS}
    models = {"det": RegressionResidual(max_iter=60).fit(train),
              "sig": Signatures(k=5, min_samples_leaf=10, lags_s=(0.5, 1.0, 2.0, 5.0), since_change=True).fit(train)}
    for kind, model in models.items():
        score = model.score if kind == "det" else model.vsd
        # whole-run
        null = NullModel(pd.DataFrame([score(r) for r in held])); thr = null.threshold("p95")
        variants = {"whole": (lambda r: score(r), thr, null)}
        for W in WINDOWS:
            tr_w = windowed_scores(model, train, W, kind)
            base_w = pd.concat(tr_w).groupby(level=0).mean()
            def rel(r, W=W, base_w=base_w):
                v = (model.score_windowed(r, W) if kind == "det" else model.vsd_windowed(r, W))
                return (v - base_w.reindex(v.index).fillna(base_w.mean())).max(axis=0)
            nullw = NullModel(pd.DataFrame([rel(r) for r in held]))
            variants[f"win{W:g}"] = (rel, nullw.threshold("p95"), nullw)
        for f, cand in cands.items():
            if len(cand) < 8:
                continue
            for vname, (fn, thr_v, null_v) in variants.items():
                rsd, fl = null_v.flag_perm(pd.DataFrame([fn(r) for r in cand]))[:2]
                rec = dict(aircraft=ac, manoeuvre=man, layer=kind, mode=vname, fault=f)
                if f == "noop":
                    rec["false_alarm"] = float(fl.mean())
                else:
                    orc = pd.Series({m: (m in b6.ORACLE[f.split("_")[0]]) for m in fl.index})
                    m = prf(fl, orc); rec.update(recall=m["recall"], precision=m["precision"], auc=auc(rsd, orc))
                rows.append(rec)
    print(ac, man, "done", flush=True)
    return rows


def main():
    aircraft = sys.argv[1:] or ["c310", "737"]
    pairs = [(ac, man) for ac in aircraft for man in BATTERY]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        results = pool.map(one_pair, pairs, chunksize=1)
    out = pd.DataFrame([r for rs in results for r in rs])
    out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "14_windowed_rq4.csv"), index=False)
    pd.set_option("display.width", 200)
    d = out[out.fault != "noop"]
    print("\n== recall / precision / AUC by layer and mode (mean over aircraft, manoeuvres, faults)")
    print(d.groupby(["layer", "mode"])[["recall", "precision", "auc"]].mean().round(3).to_string())
    print("\n== by fault (detector):")
    print(d[d.layer == "det"].pivot_table(index="mode", columns="fault", values="recall", aggfunc="mean").round(2).to_string())
    print("\n== no-op false alarms:")
    print(out[out.fault == "noop"].groupby(["layer", "mode"])["false_alarm"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
