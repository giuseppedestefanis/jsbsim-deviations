"""c310 pipeline, second pass: fixed 1800 s analysis window for every run, whole-run and windowed RSD,
take-off-phase effect sizes. Uses the cached runs from 02/03."""
import importlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
pipe = importlib.import_module("02_c310_pipeline")
from devsig.run import load_run  # noqa: E402
from devsig.signature import Signatures  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402
from devsig.evaluate import prf, to_category, auc  # noqa: E402

RUNS, ORACLE = pipe.RUNS, pipe.ORACLE_C310
T_MAX = 1800.0
TCOL = "simulation/sim-time-sec"


def load(variant, seeds):
    out = []
    for s in seeds:
        d = load_run(os.path.join(RUNS, f"{variant}_{s:04d}.parquet"))
        out.append(d[d[TCOL] <= T_MAX].reset_index(drop=True))
    return out


def report(name, flagged, rsd, thr_label, level_rows):
    oracle = pd.Series({m: bool(o[name]) for m, o in ORACLE.items()})
    idx = [m for m in oracle.index if m in flagged.index]
    m = prf(flagged[idx], oracle[idx]); c = prf(to_category(flagged[idx]), to_category(oracle[idx]))
    level_rows.append(dict(fault=name, mode=thr_label, metric_R=round(m["recall"], 2), metric_P=round(m["precision"], 2) if not np.isnan(m["precision"]) else None,
                           cat_R=round(c["recall"], 2), cat_P=round(c["precision"], 2) if not np.isnan(c["precision"]) else None,
                           auc=round(auc(rsd[idx], oracle[idx]), 2), flagged=",".join(x.split("/")[-1] for x in flagged[idx][flagged[idx]].index)))


def main():
    base = load("baseline", range(40)); train, held = base[:20], base[20:]
    cands = {f: load(f, range(100, 110)) for f in ("inertia", "lift", "pitch", "noop")}
    base_matched = load("baseline", range(100, 110))

    print("== take-off phase (first 120 s) and cruise (120-1800 s) effect size, RMS(fault-base same seed)/RMS(base i - base j)")
    key = ["position/h-sl-ft", "velocities/vc-kts", "attitude/theta-rad", "velocities/q-rad_sec", "aero/alpha-deg", "accelerations/qdot-rad_sec2", "velocities/p-rad_sec", "attitude/phi-rad"]
    for phase, lo, hi in (("takeoff", 2.0, 120.0), ("cruise", 120.0, 1800.0)):
        rows = []
        for f in ("inertia", "lift", "pitch"):
            rec = dict(phase=phase, fault=f)
            for m in key:
                df_, db_ = [], []
                for i in range(10):
                    a = cands[f][i]; b = base_matched[i]
                    ma = (a[TCOL] >= lo) & (a[TCOL] < hi); mb = (b[TCOL] >= lo) & (b[TCOL] < hi)
                    n = min(ma.sum(), mb.sum())
                    df_.append(np.sqrt(np.mean((a.loc[ma, m].to_numpy()[:n] - b.loc[mb, m].to_numpy()[:n]) ** 2)))
                    if i < 9:
                        b2 = base_matched[i + 1]; mb2 = (b2[TCOL] >= lo) & (b2[TCOL] < hi); n2 = min(mb.sum(), mb2.sum())
                        db_.append(np.sqrt(np.mean((b.loc[mb, m].to_numpy()[:n2] - b2.loc[mb2, m].to_numpy()[:n2]) ** 2)))
                rec[m.split("/")[-1]] = round(float(np.mean(df_) / (np.mean(db_) + 1e-12)), 2)
            rows.append(rec)
        print(pd.DataFrame(rows).to_string(index=False))

    rows = []
    for k in (3, 5):
        sig = Signatures(k=k).fit(train)
        null = NullModel(pd.DataFrame([sig.vsd(r) for r in held]))
        print(f"\n== K={k}: baseline VSD mean={null.vsd_base.mean():.3f}, p95 threshold mean={null.threshold('p95').mean():.3f}")
        # whole-run
        for f in ("inertia", "lift", "pitch", "noop"):
            rsd_runs = null.rsd(pd.DataFrame([sig.vsd(r) for r in cands[f]]))
            rsd, fl = flag(rsd_runs, null.threshold("p95"))
            if f == "noop":
                rows.append(dict(fault=f, mode=f"K{k} whole p95", noop_false_alarm=round(float(fl.mean()), 2)))
            else:
                report(f, fl, rsd, f"K{k} whole p95", rows)
        # windowed: per-window RSD, null threshold p99 over (held-out run, window) pairs, candidate = median over runs of max over windows
        for W in (30.0, 120.0):
            tr_w = [sig.vsd_windowed(r, W) for r in train]
            base_w = pd.concat(tr_w).groupby(level=0).mean()  # per-window training mean VSD
            def rsd_w(r):
                v = sig.vsd_windowed(r, W)
                return (v - base_w.reindex(v.index).fillna(base_w.mean()))
            null_max = pd.DataFrame([rsd_w(r).max(axis=0) for r in held])     # max over windows, per held-out run
            thr = null_max.quantile(0.95, axis=0)
            for f in ("inertia", "lift", "pitch", "noop"):
                cand_max = pd.DataFrame([rsd_w(r).max(axis=0) for r in cands[f]])
                rsd, fl = flag(cand_max, thr)
                if f == "noop":
                    rows.append(dict(fault=f, mode=f"K{k} win{W:.0f}s p95", noop_false_alarm=round(float(fl.mean()), 2)))
                else:
                    report(f, fl, rsd, f"K{k} win{W:.0f}s p95", rows)
    out = pd.DataFrame(rows)
    print("\n== Results vs initial oracle")
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "04_c310_windowed.csv"), index=False)


if __name__ == "__main__":
    main()
