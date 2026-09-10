"""RQ2: signatures (static, dynamic) vs raw-data baselines on the cached Tier A runs, same null-threshold protocol."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import importlib
import os
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
from devsig.baselines import Envelope, KS, RegressionResidual, IForest  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402
from devsig.evaluate import prf, to_category, auc  # noqa: E402

METHODS = {
    "sig_static": lambda: Signatures(k=5, min_samples_leaf=10),
    "sig_dynamic": lambda: Signatures(k=5, min_samples_leaf=10, lags_s=(0.5, 1.0, 2.0, 5.0), since_change=True),
    "envelope": lambda: Envelope(0.05),
    "ks": lambda: KS(),
    "regression": lambda: RegressionResidual(max_iter=60),
    "iforest": lambda: IForest(),
}
FAULTS = ("inertia_x1.6", "cmq_x0.5", "cmalpha_x0.7", "clo_+0.1", "noop")


def one_pair(args):
    ac, man = args
    rows = []
    base = b7.load(ac, "baseline", man, range(40))
    if len(base) < 30:
        return rows, []
    train, held = base[:len(base) // 2], base[len(base) // 2:]
    base_m = b7.load(ac, "baseline", man, range(100, 110))
    cands = {f: b7.load(ac, f, man, range(100, 110)) for f in FAULTS}
    _, outputs = __import__("devsig.signature", fromlist=["split_metrics"]).split_metrics(base[0].columns)
    ess = {f: b6.effect_size(c, base_m, outputs) if (len(c) >= 8 and len(base_m) >= 8) else None for f, c in cands.items()}
    gates = {f: (e > 1.0) if (f != "noop" and e is not None) else None for f, e in ess.items()}
    per_output = []
    for mname, factory in METHODS.items():
        model = factory().fit(train)
        score = model.vsd if hasattr(model, "vsd") else model.score
        null = NullModel(pd.DataFrame([score(r) for r in held])); thr = null.threshold("p95")
        for f, cand in cands.items():
            if len(cand) < 8:
                continue
            rsd, fl, pval = null.flag_perm(pd.DataFrame([score(r) for r in cand]))
            rec = dict(aircraft=ac, manoeuvre=man, method=mname, fault=f)
            orc_f = None if f == "noop" else pd.Series({m: (m in b6.ORACLE[f.split("_")[0]]) for m in fl.index})
            for o in fl.index:  # per-output record: score, threshold, flag, oracle membership, effect size
                per_output.append(dict(aircraft=ac, manoeuvre=man, method=mname, fault=f, output=o, rsd=float(rsd[o]), thr=float(thr[o]),
                                       flag=bool(fl[o]), p=float(pval[o]), in_oracle=(None if orc_f is None else bool(orc_f[o])),
                                       es=(None if ess.get(f) is None else float(ess[f][o]))))
            if f == "noop":
                rec.update(false_alarm=float(fl.mean()))
            else:
                orc = pd.Series({m: (m in b6.ORACLE[f.split("_")[0]]) for m in fl.index})
                m = prf(fl, orc); c = prf(to_category(fl), to_category(orc))
                rec.update(recall=m["recall"], precision=m["precision"],
                           cat_recall=c["recall"], cat_precision=c["precision"], auc=auc(rsd, orc),
                           n_flagged=int(fl.sum()), flagged=";".join(fl[fl].index))
                g = gates.get(f)
                if g is not None:
                    go = orc & g.reindex(orc.index).fillna(False)
                    rec.update(n_gated=int(go.sum()))
                    if go.sum() > 0:
                        mg = prf(fl, go)
                        rec.update(gated_recall=mg["recall"], gated_precision=mg["precision"], gated_auc=auc(rsd, go))
            rows.append(rec)
    print(ac, man, "done", flush=True)
    return rows, per_output


def main():
    aircraft = sys.argv[1:] or ["c310", "737"]
    pairs = [(ac, man) for ac in aircraft for man in BATTERY]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        results = pool.map(one_pair, pairs, chunksize=1)
    rows = [r for rs, _ in results for r in rs]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "08_baselines_rq2.csv"), index=False)
    pd.DataFrame([r for _, po in results for r in po]).to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "08_per_output.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n== mean over manoeuvres: recall / precision / AUC per method and fault")
    d = out[out.fault != "noop"]
    print(d.pivot_table(index=["aircraft", "method"], columns="fault", values=["recall", "precision", "auc"], aggfunc="mean").round(2).to_string())
    print("\n== false alarm rate on noop")
    print(out[out.fault == "noop"].groupby(["aircraft", "method"])["false_alarm"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
