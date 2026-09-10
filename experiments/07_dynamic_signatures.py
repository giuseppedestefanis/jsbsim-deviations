"""Static vs dynamic signatures on the cached Tier A runs (analysis only)."""
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
from devsig.run import load_run  # noqa: E402
from devsig.manoeuvres import BATTERY  # noqa: E402
from devsig.signature import Signatures  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402
from devsig.evaluate import prf, to_category, auc  # noqa: E402

RUNS = b6.RUNS
CONFIGS = {
    "static": dict(k=5, min_samples_leaf=10),
    "lags": dict(k=5, min_samples_leaf=10, lags_s=(0.5, 1.0, 2.0, 5.0)),
    "lags+since": dict(k=5, min_samples_leaf=10, lags_s=(0.5, 1.0, 2.0, 5.0), since_change=True),
    "since": dict(k=5, min_samples_leaf=10, since_change=True),
}


def load(ac, variant, man, seeds):
    out = []
    for s in seeds:
        p = os.path.join(RUNS, ac, f"{variant}_{man}_{s:04d}.parquet")
        if os.path.exists(p):
            out.append(load_run(p))
    return out


def one_pair(args):
    ac, man = args
    rows = []
    base = load(ac, "baseline", man, range(40))
    if len(base) < 30:
        return rows
    train, held = base[:len(base) // 2], base[len(base) // 2:]
    cands = {f: load(ac, f, man, range(100, 110)) for f in ("inertia_x1.6", "cmq_x0.5", "cmalpha_x0.7", "clo_+0.1", "noop")}
    for cname, kw in CONFIGS.items():
        sig = Signatures(**kw).fit(train)
        null = NullModel(pd.DataFrame([sig.vsd(r) for r in held])); thr = null.threshold("p95")
        rec = dict(aircraft=ac, manoeuvre=man, config=cname, base_vsd=round(float(null.vsd_base.mean()), 3), thr=round(float(thr.mean()), 3))
        for f, cand in cands.items():
            if len(cand) < 8:
                rec[f] = "n/a"; continue
            rsd, fl = null.flag_perm(pd.DataFrame([sig.vsd(r) for r in cand]))[:2]
            if f == "noop":
                rec["noop_fa"] = round(float(fl.mean()), 2); continue
            orc = pd.Series({m: (m in b6.ORACLE[f.split("_")[0]]) for m in sig.outputs})
            m = prf(fl, orc); c = prf(to_category(fl), to_category(orc))
            rec[f] = f"{m['recall']:.2f}/{(0 if np.isnan(m['precision']) else m['precision']):.2f} cat {c['recall']:.2f} auc {auc(rsd, orc):.2f}"
        rows.append(rec)
    print(ac, man, "done", flush=True)
    return rows


def main():
    aircraft = sys.argv[1:] or ["c310", "737"]
    pairs = [(ac, man) for ac in aircraft for man in BATTERY]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        results = pool.map(one_pair, pairs, chunksize=1)
    out = pd.DataFrame([r for rs in results for r in rs])
    out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "07_dynamic_signatures.csv"), index=False)
    print("\n== mean over pairs per aircraft/config: base_vsd and metric-level recall per fault")
    def rec_of(s):
        return float(s.split("/")[0]) if isinstance(s, str) and "/" in s and s != "n/a" else np.nan
    summ = out.assign(**{f + "_R": out[f].map(rec_of) for f in ("inertia_x1.6", "cmq_x0.5", "cmalpha_x0.7", "clo_+0.1") if f in out})
    cols = ["base_vsd", "noop_fa"] + [c for c in summ.columns if c.endswith("_R")]
    print(summ.groupby(["config"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
